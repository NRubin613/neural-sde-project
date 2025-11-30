import numpy as np
import torch

class UniversalFeatureEngine:
    def __init__(self, decay_span=20):
        # Decay factor for EMA (Smoothing)
        self.alpha = 2 / (decay_span + 1)
        
        # Internal states for recursion
        self.prev_price = None
        self.ema_var = 0.0      # Variance state
        self.ema_trend = 0.0    # Trend state
        self.avg_gain = 0.0     # RSI state
        self.avg_loss = 0.0     # RSI state

        # Normalization Stats (Learned during fit)
        self.feature_means = None
        self.feature_stds = None

    def prime_engine(self, price_history):
        """
        WARMUP: Feeds historical prices to settle EMAs/RSI states.
        """
        self.prev_price = None # Reset
        for price in price_history:
            self._step_logic(price)
            
    def fit_transform(self, prices):
        """
        BATCH MODE: Prepares training data from a price array.
        Returns: Normalized Features (Tensor), Targets (Tensor)
        """
        prices = np.array(prices)
        n = len(prices)
        
        feat_log_ret = np.zeros(n)
        feat_vol = np.zeros(n)
        feat_trend = np.zeros(n)
        feat_rsi = np.zeros(n)
        
        # Reset state
        self.prev_price = None
        
        for i in range(n):
            f = self._step_logic(prices[i])
            feat_log_ret[i], feat_vol[i], feat_trend[i], feat_rsi[i] = f

        # Interaction Term
        interaction = feat_vol * feat_trend
        
        # Stack raw features: [LogRet, Vol, Trend, RSI, Interaction]
        X_raw = np.column_stack((feat_log_ret, feat_vol, feat_trend, feat_rsi, interaction))
        
        # Calculate Normalization Stats (ignoring warmup period)
        warmup = 50
        self.feature_means = X_raw[warmup:].mean(axis=0)
        self.feature_stds = X_raw[warmup:].std(axis=0) + 1e-6
        
        # Normalize
        X_norm = (X_raw - self.feature_means) / self.feature_stds
        
        # Targets: Next day's log-return
        # y[i] is the return that happens at i+1
        y = np.roll(feat_log_ret, -1)
        
        # Return sliced data (remove warmup and last NaN target)
        return (
            torch.FloatTensor(X_norm[warmup:-1]), 
            torch.FloatTensor(y[warmup:-1]).unsqueeze(-1)
        )

    def update_simulation(self, new_price):
        """
        ONLINE MODE: Updates state and returns NORMALIZED feature vector for NN.
        """
        features = self._step_logic(new_price)
        
        # Add interaction
        interaction = features[1] * features[2]
        full_feat = np.append(features, interaction)
        
        # Normalize using learned stats
        if self.feature_means is None:
            raise ValueError("Engine must be fit on data before simulation!")
            
        norm_feat = (full_feat - self.feature_means) / self.feature_stds
        return torch.FloatTensor(norm_feat)

    def _step_logic(self, price):
        # Handle Cold Start
        if self.prev_price is None:
            self.prev_price = price
            self.ema_trend = price
            self.ema_var = 0.0001
            return np.array([0.0, 0.0, 0.0, 0.5]) 

        # 1. Log Return
        ret = np.log(price / (self.prev_price + 1e-8))
        
        # 2. Volatility (EWMA of Squared Returns)
        self.ema_var = (self.alpha * ret**2) + ((1 - self.alpha) * self.ema_var)
        vol = np.sqrt(self.ema_var)
        
        # 3. Trend (EMA Distance)
        self.ema_trend = (self.alpha * price) + ((1 - self.alpha) * self.ema_trend)
        trend_dist = (price - self.ema_trend) / (self.ema_trend + 1e-8)
        
        # 4. RSI (Wilder's Smoothing)
        change = price - self.prev_price
        gain = max(change, 0)
        loss = max(-change, 0)
        rsi_alpha = 1/14
        self.avg_gain = (self.avg_gain * (1 - rsi_alpha)) + gain
        self.avg_loss = (self.avg_loss * (1 - rsi_alpha)) + loss
        
        if self.avg_loss == 0:
            rsi = 1.0
        else:
            rs = self.avg_gain / self.avg_loss
            rsi = 1.0 - (1.0 / (1.0 + rs))
            
        self.prev_price = price
        return np.array([ret, vol, trend_dist, rsi])