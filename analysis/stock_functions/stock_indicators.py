import pandas as pd
from talib import abstract as ta

from alias.alias import Alias, aliased
from helper_functions.general_helpers import check_matype, check_series_type


@aliased
class StockIndicatorCalculator:

    # Basic Averages

    @Alias('sma', 'SMA')
    def simple_moving_average(self, data, num_periods: int = 3, series_type: str = 'close') -> pd.Series:
        """
        Moving Averages are used to smooth the data in an array to help eliminate noise and identify trends. The
        Simple Moving Average is literally the simplest form of a moving average. Each output value is the average of
        the previous n values. In a Simple Moving Average, each value in the time period carries equal weight,
        and values outside of the time period are not included in the average. This makes it less responsive to
        recent changes in the data, which can be useful for filtering out those changes.

        :param data: Data to calculate over
        :param num_periods: specifies the number of periods for each calculation
        :param series_type: the price data to calculate over
        :return: returns a pandas Series with the simple moving average
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        series_type = check_series_type(series_type)
        return ta.SMA(data, timeperiod=num_periods, price=series_type)

    @Alias('ema', 'EMA')
    def exponential_moving_average(self, data, num_periods: int = 3, series_type: str = 'close') -> pd.Series:
        """
        The Exponential Moving Average is a staple of technical analysis and is used in countless technical
        indicators. In a Simple Moving Average, each value in the time period carries equal weight, and values
        outside of the time period are not included in the average. However, the Exponential Moving Average is a
        cumulative calculation, including all data. Past values have a diminishing contribution to the average,
        while more recent values have a greater contribution. This method allows the moving average to be more
        responsive to changes in the data.

        :param data: Data to calculate over
        :param num_periods: specifies the number of periods for each calculation
        :param series_type: the price data to calculate over
        :return: returns a pandas Series with the exponential moving average
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        series_type = check_series_type(series_type)
        return ta.EMA(data, timeperiod=num_periods, price=series_type)

    @Alias('vwap', 'VWAP')
    def volume_weighted_average_price(self, data) -> pd.Series:
        """
        The daily price times volume summed divided by the total volume of the period. Essentially a weighted average
        of price with the weights being the volume of that time period.

        :param data: Data to calculate over
        :return: volume weighted average price for the entire time period of historical data
        :rtype: pandas.Series
        """
        cols = ['High', 'Low', 'Close']
        typical_price = data.loc[:, cols].sum(axis=1).div(3)
        vwap = typical_price * data.loc[:, 'Volume']
        return pd.Series(vwap.values, data.index, name='VWAP')

    @Alias('wma', 'WMA')
    def weighted_moving_average(self, data, num_periods: int = 10, series_type: str = 'close') -> pd.Series:
        """
        The Weighted Moving Average calculates a weight for each value in the series. The more recent values are
        assigned greater weights. The Weighted Moving Average is similar to a Simple Moving average in that it is not
        cumulative, that is, it only includes values in the time period (unlike an Exponential Moving Average). The
        Weighted Moving Average is similar to an Exponential Moving Average in that more recent data has a greater
        contribution to the average.

        :param data: Data to calculate over
        :param num_periods: specifies the number of periods for each calculation
        :param series_type: the price data to calculate over
        :return: returns a pandas Series with the weighted moving average
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        series_type = check_series_type(series_type)
        return ta.WMA(data, timeperiod=num_periods, price=series_type)

    @Alias('dema', 'DEMA')
    def double_exponential_moving_average(self, data, num_periods: int = 10, series_type: str = 'close') -> pd.Series:
        """
        The DEMA is a smoothing indicator with less lag than a straight exponential moving average. DEMA is an
        acronym for Double Exponential Moving Average, but the calculation is more complex than just a moving average
        of a moving average.

        :param data: Data to calculate over
        :param num_periods: specifies the number of periods for each calculation
        :param series_type: the price data to calculate over
        :return: returns a pandas Series with the double exponential moving average
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        series_type = check_series_type(series_type)
        return ta.DEMA(data, timeperiod=num_periods, price=series_type)

    @Alias('tema', 'TEMA', 't3', 'T3')
    def triple_exponential_moving_average(self, data, num_periods: int = 5, vfactor: float = 0.7,
                                          series_type: str = 'close') -> pd.Series:
        """
        The TEMA is a smoothing indicator with less lag than a straight exponential moving average. TEMA is an
        acronym for Triple Exponential Moving Average, but the calculation is more complex than that.

        The T3 is a type of moving average, or smoothing function. It is based on the DEMA. The T3 takes the DEMA
        calculation and adds a vfactor which is between zero and 1. The resultant function is called the GD,
        or Generalized DEMA. A GD with vfactor of 1 is the same as the DEMA. A GD with a vfactor of zero is the same
        as an Exponential Moving Average. The T3 typically uses a vfactor of 0.7.

        :param data: Data to calculate over
        :param num_periods: specifies the number of periods for each calculation
        :param vfactor: a value that indicates the smoothing of the function, between 0 and 1. If 1, then TEMA = DEMA
        while if 0, TEMA = EMA
        :param series_type: the price data to calculate over
        :return: returns a pandas Series with the triple exponential moving average
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        assert 0. <= vfactor <= 1., 'vfactor must be between 0 and 1 inclusive'
        series_type = check_series_type(series_type)
        return ta.TEMA(data, timeperiod=num_periods, vfactor=vfactor, price=series_type)

    @Alias('trima', 'TRIMA')
    def triangular_moving_average(self, data, num_periods: int = 10, series_type: str = 'close') -> pd.Series:
        """
        A form of weighted moving average that puts more weight on the middle of the time frame and less on the ends.

        :param data: Data to calculate over
        :param num_periods: specifies the number of periods for each calculation
        :param series_type: the price data to calculate over
        :return: returns a pandas Series with the triangular moving average
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        series_type = check_series_type(series_type)
        return ta.TRIMA(data, timeperiod=num_periods, price=series_type)

    @Alias('kama', 'KAMA')
    def kaufman_adaptive_moving_average(self, data, num_periods: int = 10, series_type: str = 'close') -> pd.Series:
        """
        A moving average designed to account for market noise/volatility

        :param data: Data to calculate over
        :param num_periods: specifies the number of periods for each calculation
        :param series_type: the price data to calculate over
        :return: returns a pandas Series with the kaufman adaptive moving average
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        series_type = check_series_type(series_type)
        return ta.KAMA(data, timeperiod=num_periods, price=series_type)

    @Alias('mama', 'MAMA', 'mesa', 'MESA')
    def mesa_adaptive_moving_average(self, data, fast: float = 0.01, slow: float = 0.01,
                                     series_type: str = 'close') -> pd.DataFrame:
        """
        Trend following indicator based on the rate change of phase as measured by the Hilbert Transform Discriminator.
        Uses a fast and slow moving average to quickly respond to price changes.

        :param data: Data to calculate over
        :param fast: specifies the number of periods for each fast moving average calculation
        :param slow: specifies the number of periods for each slow moving average calculation
        :param series_type: the price data to calculate over
        :return: returns a pandas DataFrame with the mesa adaptive moving average: first column is "mama",
        second column is "fama"
        :rtype: pandas.DataFrame
        """
        assert fast >= 0, 'fast must be greater than or equal to 0'
        assert slow >= 0, 'slow must be greater than or equal to 0'
        series_type = check_series_type(series_type)
        return ta.MAMA(data, fastlimit=fast, slowlimit=slow, price=series_type)

    @Alias('macd', 'MACD')
    def moving_average_convergence_divergence(self, data, slow: int = 26, fast: int = 12, signal: int = 9,
                                              series_type: str = 'close') -> pd.DataFrame:
        """
        The MACD is the difference between two exponential moving averages with a signal line being the exponential
        moving average of the MACD. Signals trend changes and indicates new trends. High values indicate overbought
        conditions, low values indicate oversold conditions. Divergence with the price indicates an end to the current
        trend, especially if the MACD is at extreme high or low values. When the MACD line crosses above the signal
        line a buy signal is generated. When the MACD crosses below the signal line a sell signal is generated. To
        confirm the signal, the MACD should be above zero for a buy, and below zero for a sell.

        *Typically buy when MACD is above the signal line and short when MACD is below the signal line

        :param data: Data to calculate over
        :param slow: specifies the number of periods for each slow exponential moving average calculation
        :param fast: specifies the number of periods for each fast exponential moving average calculation
        :param signal: specifies the number of periods for the signal line exponential moving average calculation
        :param series_type: the price data to calculate over
        :return: returns a pandas DataFrame with the moving average convergence/divergence: first column is "macd",
        second column is "macdsignal", third column is "macdhist"
        :rtype: pandas.DataFrame
        """
        assert fast > 0, 'Short period must be greater than 0'
        assert slow > fast, 'Long period must be greater than 0'
        assert signal > 0, 'signal must be greater than 0'
        series_type = check_series_type(series_type)
        return ta.MACD(data, price=series_type, fastperiod=fast, slowperiod=slow, signal_period=signal)

    @Alias('macdext', 'MACDEXT')
    def moving_average_convergence_divergence_matype(self, data, slow: int = 26, slow_matype: int = 1, fast: int = 12,
                                                     fast_matype: int = 1, signal: int = 9, signal_matype: int = 1,
                                                     series_type: str = 'close') -> pd.DataFrame:
        """
        The MACDEXT is the difference between two arbitrary moving averages with a signal line being the arbitrary
        moving average of the MACDEXT. Signals trend changes and indicates new trends. High values indicate overbought
        conditions, low values indicate oversold conditions. Divergence with the price indicates an end to the current
        trend, especially if the MACD is at extreme high or low values. When the MACD line crosses above the signal
        line a buy signal is generated. When the MACD crosses below the signal line a sell signal is generated. To
        confirm the signal, the MACDEXT should be above zero for a buy, and below zero for a sell.

        *Typically buy when MACD is above the signal line and short when MACD is below the signal line

        :param data: Data to calculate over
        :param slow: specifies the number of periods for each slow arbitrary moving average calculation
        :param slow_matype: matype to use for the slow arbitrary moving average
        :param fast: specifies the number of periods for each fast arbitrary moving average calculation
        :param fast_matype: matype to use for the fast arbitrary moving average
        :param signal: specifies the number of periods for the signal line arbitrary moving average calculation
        :param signal_matype: matype to use for the signal arbitrary moving average
        :param series_type: the price data to calculate over
        :return: returns a pandas DataFrame with the moving average convergence/divergence: first column is "macd",
        second
        column is "macdsignal", third column is "macdhist"
        :rtype: pandas.DataFrame
        """
        assert fast > 0, 'fast must be greater than 0'
        assert slow > fast, 'slow must be greater than fast'
        assert signal > 0, 'signal must be greater than 0'
        slow_matype = check_matype(slow_matype, 'slow_matype')
        fast_matype = check_matype(fast_matype, 'fast_matype')
        signal_matype = check_matype(signal_matype, 'signal_matype')
        series_type = check_series_type(series_type)
        return ta.MACDEXT(data, price=series_type, fastperiod=fast, fastmatype=fast_matype,
                          slowperiod=slow,
                          slowmatype=slow_matype, signalperiod=signal, signalmatype=signal_matype)

    # Other Indicators

    @Alias('stoch', 'STOCH', 'stoch_oscillator')
    def stochastic_oscillator(self, data, fast_k_period: int = 5, slow_k_period: int = 3, slow_d_period: int = 3,
                              slow_k_ma_type: int = 1, slow_d_ma_type: int = 1) -> pd.DataFrame:
        """
        The Stochastic Oscillator measures where the close is in relation to the recent trading range. The values range
        from zero to 100. %D values over 75 indicate an overbought condition; values under 25 indicate an oversold
        condition. When the Fast %D crosses above the Slow %D, it is a buy signal; when it crosses below, it is a sell
        signal. The Raw %K is generally considered too erratic to use for crossover signals. %K is the original
        calculation and %D takes the moving average of %K, both values are slow which refers to them being smoothed.

        *Bound between 0 and 100, used to dictate over bought/sold positions. 80+ indicates over bought, 20- indicates
        over sold. Should look for changes in this to help with future changes, not helping with long term trends.
        Stochastic oscillator charting generally consists of two lines: one reflecting the actual value of the
        oscillator for each session, and one reflecting its three-day simple moving average. Because price is thought
        to follow momentum, intersection of these two lines is considered to be a signal that a reversal may be in
        the works, as it indicates a large shift in momentum from day to day. Divergence between the stochastic
        oscillator and trending price action is also seen as an important reversal signal. For example,
        when a bearish trend reaches a new lower low, but the oscillator prints a higher low, it may be an indicator
        that bears are exhausting their momentum and a bullish reversal is brewing. Transaction signals are usually
        made when the %K crosses through the %D.

        :param data: Data to calculate over
        :param fast_k_period: specifies the number of periods for each fast k arbitrary moving average calculation
        :param slow_k_period: specifies the number of periods for each slow k arbitrary moving average calculation
        :param slow_d_period: specifies the number of periods for each slow d arbitrary moving average calculation
        :param slow_k_ma_type: specifies the arbitrary moving average function for each slow k calculation
        :param slow_d_ma_type: specifies the arbitrary moving average function for each slow d calculation
        :return: returns a pandas DataFrame with the slow % values where slow is referring to using the smoothed %K:
        first column is "slowk", second column is "slowd"
        :rtype: pandas.DataFrame
        """
        assert slow_k_period > 0, 'slow_k_period must be greater than 0'
        assert fast_k_period > slow_k_period, 'fast_k_period must be greater than slow_k_period'
        assert slow_d_period > 0, 'slow_d_period must be greater than 0'
        slow_k_ma_type = check_matype(slow_k_ma_type, 'slow_k_ma_type')
        slow_d_ma_type = check_matype(slow_d_ma_type, 'slow_d_ma_type')
        return ta.STOCH(data, fastk_period=fast_k_period, slowk_period=slow_k_period,
                        slowk_matype=slow_k_ma_type, slowd_period=slow_d_period, slowd_matype=slow_d_ma_type)

    @Alias('stochf', 'STOCHF')
    def stochastic_fast(self, data, fast_k_period: int = 5, fast_d_period: int = 3, matype: int = 1) -> pd.DataFrame:
        """
        The Fast Stochastic Oscillator measures where the close is in relation to the recent trading range. The values
        range from zero to 100. %D values over 75 indicate an overbought condition; values under 25 indicate an
        oversold
        condition. When the Fast %D crosses above the Slow %D, it is a buy signal; when it crosses below, it is a sell
        signal. The Raw %K is generally considered too erratic to use for crossover signals. %K is the original
        calculation and %D takes the moving average of %K, both values are fast which refers to them being un-smoothed.

        *Bound between 0 and 100, used to dictate over bought/sold positions. 80+ indicates over bought, 20- indicates
        over sold. Should look for changes in this to help with future changes, not helping with long term trends.
        Stochastic oscillator charting generally consists of two lines: one reflecting the actual value of the
        oscillator for each session, and one reflecting its three-day simple moving average. Because price is thought
        to follow momentum, intersection of these two lines is considered to be a signal that a reversal may be in
        the works, as it indicates a large shift in momentum from day to day. Divergence between the stochastic
        oscillator and trending price action is also seen as an important reversal signal. For example,
        when a bearish trend reaches a new lower low, but the oscillator prints a higher low, it may be an indicator
        that bears are exhausting their momentum and a bullish reversal is brewing. Transaction signals are usually
        made when the %K crosses through the %D.

        :param data: Data to calculate over
        :param fast_k_period: specifies the number of periods for each slow k arbitrary moving average calculation
        :param fast_d_period: specifies the number of periods for each fast k arbitrary moving average calculation
        :param matype: specifies the arbitrary moving average function for each fast d calculation
        :return: returns a pandas DataFrame with the fast % values where fast is referring to using the un-smoothed %K:
        first column is "fastk", second column is "fastd"
        :rtype: pandas.DataFrame
        """
        assert fast_k_period > 0, 'fast_k_period must be greater than 0'
        assert fast_d_period > 0, 'fast_d_period must be greater than 0'
        matype = check_matype(matype, 'matype')
        return ta.STOCHF(data, fastk_period=fast_k_period, fastd_period=fast_d_period, fastd_matype=matype)

    @Alias('stochrsi', 'STOCHRSI')
    def stochastic_relative_strength_index(self, data, num_periods: int = 14, series_type: str = 'close',
                                           fast_k_period: int = 5, fast_d_period: int = 3,
                                           matype: int = 1) -> pd.DataFrame:
        """
        Stochastic RSI (StochRSI) is an indicator of an indicator. It calculates the RSI relative to its range in order
        to increase the sensitivity of the standard RSI. The values of the StochRSI are from zero to one.

        The Stochastic RSI can be interpreted several ways. Overbought/oversold conditions are indicated when the
        StochRSI crosses above .20 / below .80. A buy signal is generated when the StochRSI moves from oversold to
        above the midpoint (.50). A sell signal is generated when the StochRSI moves from overbought to below the
        midpoint. Also look for divergence with the price to indicate the end of a trend.

        *Over 0.8 is over bought, under 0.2 is over sold

        :param data: Data to calculate over
        :param num_periods: The number of periods to calculate over
        :param series_type: The pricing data to use for the calculation
        :param fast_k_period: The number of periods to use for calculating fast (un-smoothed) %K
        :param fast_d_period: The number of periods to use for calculating fast (un-smoothed) %D
        :param matype: The arbitrary function to use as a moving average
        :return: Returns a pandas DataFrame of fast (un-smoothed) values: first column is "fastk", second column is
        "fastd"
        :rtype: pandas.DataFrame
        """
        matype = check_matype(matype, 'matype')
        return ta.STOCHRSI(data, timeperiod=num_periods, price=series_type, fastk_period=fast_k_period,
                           fastd_period=fast_d_period, fastd_matype=matype)

    @Alias('rsi', 'RSI', 'relative_strength')
    def relative_strength_index(self, data, num_periods: int = 5, series_type: str = 'close') -> pd.Series:
        """
        The Relative Strength Index (RSI) calculates a ratio of the recent upward price movements to the absolute price
        movement. The RSI ranges from 0 to 100. The RSI is interpreted as an overbought/oversold indicator when the
        value is over 70/below 30. You can also look for divergence with price. If the price is making new highs/lows,
        and the RSI is not, it indicates a reversal.

        *Traditional interpretation and usage of the RSI are that values of 70 or above indicate that a security is
        becoming overbought or overvalued and may be primed for a trend reversal or corrective pullback in price. An
        RSI reading of 30 or below indicates an oversold or undervalued condition.

        :param data: Data to calculate over
        :param num_periods: The number of periods for each rolling calculation
        :param series_type: The pricing data to calculate over
        :return: Returns a pandas Series of calculated values
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        series_type = check_series_type(series_type)
        return ta.RSI(data, timeperiod=num_periods, price=series_type)

    @Alias('willr', 'WILLR')
    def williams_r(self, data, num_periods: int = 14) -> pd.Series:
        """
        The Williams %R is similar to an un-smoothed Stochastic %K. The values range from zero to 100, and are charted
        on an inverted scale, that is, with zero at the top and 100 at the bottom. Values below 20 indicate an
        overbought condition and a sell signal is generated when it crosses the 20 line. Values over 80 indicate an
        oversold condition and a buy signal is generated when it crosses the 80 line.

        *When the indicator is between 0 and 20 the price is overbought, or near the high of its recent price
        range. When the indicator is between 80 and 100 the price is oversold, or far from the high of its recent
        range. During an uptrend, traders can watch for the indicator to move above 80. When the price starts moving
        up, and the indicator moves back below 80, it could signal that the uptrend in price is starting again.The
        same concept could be used to find short trades in a downtrend. When the indicator is below 20, watch for the
        price to start falling along with the Williams %R moving back above 20 to signal a potential continuation of
        the downtrend. Traders can also watch for momentum failures. During a strong uptrend, the price will often
        reach 20 or below. If the indicator falls, and then can't get back below 20 before falling again,
        that signals that the upward price momentum is in trouble and a bigger price decline could follow. The same
        concept applies to a downtrend. Readings of 80 or above are often reached. When the indicator can no longer
        reach those low levels before moving higher it could indicate the price is going to head higher.

        :param data: Data to calculate over
        :param num_periods: The number of periods to perform the rolling calculation over
        :return: Returns a pandas Series of calculated values
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        return ta.WILLR(data, timeperiod=num_periods)

    @Alias('atr', 'ATR')
    def average_true_range(self, data, num_periods: int = 14) -> pd.Series:
        """
        The ATR is a Welles Wilder style moving average of the True Range. The ATR is a measure of volatility. High ATR
        values indicate high volatility, and low values indicate low volatility, often seen when the price is flat.

        The ATR is a component of the Welles Wilder Directional Movement indicators (+/-DI, DX, ADX and ADXR).

        *Measures volatility (place a sell order at n times the ATR below the highest high since you entered a position.

        :param data: Data to calculate over
        :param num_periods: The number of periods to use for the rolling calculation
        :return: Returns a pandas Series of calculated values
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        return ta.ATR(data, timeperiod=num_periods)

    @Alias('adx', 'ADX', 'average_directional_movement')
    def average_directional_movement_index(self, data, num_periods: int = 14) -> pd.Series:
        """
        The ADX is a Welles Wilder style moving average of the Directional Movement Index (DX). The values range from 0
        to 100, but rarely get above 60. To interpret the ADX, consider a high number to be a strong trend, and a low
        number, a weak trend.

        *Quantifies trend strength, over 25 is considered trending

        :param data: Data to calculate over
        :param num_periods: The number of periods to perform the rolling calculation over
        :return: Returns a pandas Series of calculated values
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        return ta.ADX(data, timeperiod=num_periods)

    @Alias('adxr', 'ADXR')
    def average_directional_movement_index_rating(self, data, num_periods: int = 10) -> pd.Series:
        """
        The ADXR is equal to the current ADX plus the ADX from n bars ago divided by 2. In effect, it is the average of
        the two ADX values. The ADXR smooths the ADX, and is therefore less responsive, however, the ADXR filters out
        excessive tops and bottoms. To interpret the ADXR, consider a high number to be a strong trend, and a low
        number, a weak trend.

        *Essentially the ADX but can be used as a more conservative estimate. Can be used by buying when ADX > ADXR
        and selling when ADX < ADXR when at least the ADX < 25.

        :param data: Data to calculate over
        :param num_periods: The number of periods to perform the rolling calculation over
        :return: Returns a pandas Series of calculated values
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        return ta.ADXR(data, timeperiod=num_periods)

    @Alias('apo', 'APO')
    def absolute_price_oscillator(self, data, series_type: str = 'close', fast: int = 12, slow: int = 26,
                                  matype: int = 0) -> pd.Series:
        """
        The Price Oscillator shows the difference between two moving averages. It is basically a MACD, but the Price
        Oscillator can use any time periods. A buy signal is generate when the Price Oscillator rises above zero,
        and a sell signal when the it falls below zero.

        *Returned numbers indicate the amount the short average is above/below the long average

        :param data: Data to calculate over
        :param series_type: The pricing data to perform the rolling calculation over
        :param fast: The period to perform the fast arbitrary moving average over
        :param slow: The period to perform the slow arbitrary moving average over
        :param matype: The arbitrary moving average to use for calculations
        :return: Returns a pandas Series with calculated values
        :rtype: pandas.Series
        """
        assert fast > 0, 'fast must be greater than 0'
        assert slow > fast, 'slow must be greater than fast'
        series_type = check_series_type(series_type)
        matype = check_matype(matype, 'matype')
        return ta.APO(data, price=series_type, fastperiod=fast, slowperiod=slow, matype=matype)

    @Alias('ppo', 'PPO')
    def percentage_price_oscillator(self, data, series_type: str = 'close', fast: int = 12, slow: int = 26,
                                    matype: int = 0) -> pd.Series:
        """
        The Price Oscillator Percent shows the percentage difference between two moving averages. A buy signal is
        generate when the Price Oscillator Percent rises above zero, and a sell signal when the it falls below zero.

        *Returned numbers indicate the percent that the short term average is greater/less than the long term average.

        :param data: Data to calculate over
        :param series_type: The pricing data to perform the rolling calculation over
        :param fast: The period to perform the fast arbitrary moving average over
        :param slow: The period to perform the slow arbitrary moving average over
        :param matype: The arbitrary moving average to use for calculations
        :return: Returns a pandas Series with calculated values
        :rtype: pandas.Series
        """
        assert fast > 0, 'fast must be greater than 0'
        assert slow > fast, 'slow must be greater than fast'
        series_type = check_series_type(series_type)
        matype = check_matype(matype, 'matype')
        return ta.PPO(data, price=series_type, fastperiod=fast, slowperiod=slow, matype=matype)

    @Alias('mom', 'MOM')
    def momentum(self, data, num_periods: int = 10, series_type: str = 'close') -> pd.Series:
        """
        The Momentum is a measurement of the acceleration and deceleration of prices. It indicates if prices are
        increasing at an increasing rate or decreasing at a decreasing rate. The Momentum function can be applied to
        the price, or to any other data series.

        *When applied, an investor can buy or sell based on the strength of the trends in an asset's price. If a
        trader wants to use a momentum-based strategy, he takes a long position in a stock or asset that has been
        trending up. If the stock is trending down, he takes a short position. Instead of the traditional philosophy
        of trading—buy low, sell high—momentum investing seeks to sell low and buy lower, or buy high and sell
        higher. Instead of identifying the continuation or reversal pattern, momentum investors focus on the trend
        created by the most recent price break.

        :param data: Data to calculate over
        :param num_periods: The number of periods for each rolling calculation
        :param series_type: The pricing data to calculate over
        :return: Returns a pandas Series of calculated values
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        series_type = check_series_type(series_type)
        return ta.MOM(data, timeperiod=num_periods, price=series_type)

    @Alias('bop', 'BOP')
    def balance_of_power(self, data) -> pd.Series:
        """
        The Balance of Power indicator measures the market strength of buyers against sellers by assessing the
        ability of each side to drive prices to an extreme level. The calculation is: Balance of Power = (Close price –
        Open price) / (High price – Low price) The resulting value can be smoothed by a moving average.

        *The BOP oscillates around zero center line in the range from -1 to +1. Positive BOP reading is an indication
        of buyers' dominance and negative BOP reading is a sign of the stronger selling pressure. When BOP is equal
        zero it indicates that buyers and sellers are equally strong.

        :param data: Data to calculate over
        :return: Returns a pandas Series of calculated values
        :rtype: pandas.Series
        """
        return ta.BOP(data)

    @Alias('cci', 'CCI', 'commodity_channel')
    def commodity_channel_index(self, data, num_periods: int = 20) -> pd.Series:
        """
        The CCI is designed to detect beginning and ending market trends. The range of 100 to -100 is the normal
        trading range. CCI values outside of this range indicate overbought or oversold conditions. You can also look
        for price divergence in the CCI. If the price is making new highs, and the CCI is not, then a price
        correction is likely.

        *The CCI compares the current price to an average price over a period of time. The indicator fluctuates above
        or below zero, moving into positive or negative territory. While most values, approximately 75%, fall between
        -100 and +100, about 25% of the values fall outside this range, indicating a lot of weakness or strength in
        the price movement. When the CCI is above +100, this means the price is well above the average price as
        measured by the indicator. When the indicator is below -100, the price is well below the average price.

        :param data: Data to calculate over
        :param num_periods: The number of periods to perform the rolling calculation over
        :return: Returns a pandas Series of calculated values
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        return ta.CCI(data, timeperiod=num_periods)

    @Alias('cmo', 'CMO')
    def chande_momentum_oscillator(self, data, num_periods: int = 14, series_type: str = 'close') -> pd.Series:
        """
        The Chande Momentum Oscillator is a modified RSI. Where the RSI divides the upward movement by the net
        movement (up / (up + down)), the CMO divides the total movement by the net movement ((up - down) / (up + down)).

        *There are several ways to interpret the CMO. Values over 50 indicate overbought conditions, while values
        under -50 indicate oversold conditions. High CMO values indicate strong trends. When the CMO crosses above a
        moving average of the CMO, it is a buy signal, crossing down is a sell signal.

        :param data: Data to calculate over
        :param num_periods: The number of periods for each rolling calculation
        :param series_type: The pricing data to calculate over
        :return: Returns a pandas Series of calculated values
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        series_type = check_series_type(series_type)
        return ta.CMO(data, timeperiod=num_periods, price=series_type)

    @Alias('roc', 'ROC')
    def rate_of_change(self, data, num_periods: int = 10, series_type: str = 'close') -> pd.Series:
        """
        The Rate of Change function measures rate of change relative to previous periods. The function is used to
        determine how rapidly the data is changing. The factor is usually 100, and is used merely to make the numbers
        easier to interpret or graph. The function can be used to measure the Rate of Change of any data series,
        such as price or another indicator. When used with the price, it is referred to as the Price Rate Of Change,
        or PROC.

        :param data: Data to calculate over
        :param num_periods: The number of periods for each rolling calculation
        :param series_type: The pricing data to calculate over
        :return: Returns a pandas Series of calculated values
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        series_type = check_series_type(series_type)
        return ta.ROC(data, timeperiod=num_periods, price=series_type)

    @Alias('rocr', 'ROCR')
    def rate_of_change_ratio(self, data, num_periods: int = 10, series_type: str = 'close') -> pd.Series:
        """
        The ratio of the Rate of Change function measures rate of change relative to previous periods. The function is
        used to determine how rapidly the data is changing. The factor is usually 100, and is used merely to make the
        numbers easier to interpret or graph. The function can be used to measure the Rate of Change of any data series,
        such as price or another indicator. When used with the price, it is referred to as the Price Rate Of Change,
        or PROC.

        :param data: Data to calculate over
        :param num_periods: The number of periods for each rolling calculation
        :param series_type: The pricing data to calculate over
        :return: Returns a pandas Series of calculated values
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        series_type = check_series_type(series_type)
        return ta.ROCR(data, timeperiod=num_periods, price=series_type)

    @Alias('Aroon', 'AROON')
    def aroon(self, data, num_periods: int = 25) -> pd.DataFrame:
        """
        The word aroon is Sanskrit for "dawn's early light." The Aroon indicator attempts to show when a new trend is
        dawning. The indicator consists of two lines (Up and Down) that measure how long it has been since the
        highest high/lowest low has occurred within an n period range.

        *When the Aroon Up is staying between 70 and 100 then it indicates an upward trend. When the Aroon Down is
        staying between 70 and 100 then it indicates an downward trend. A strong upward trend is indicated when the
        Aroon Up is above 70 while the Aroon Down is below 30. Likewise, a strong downward trend is indicated when
        the Aroon Down is above 70 while the Aroon Up is below 30. Also look for crossovers. When the Aroon Down
        crosses above the Aroon Up, it indicates a weakening of the upward trend (and vice versa).

        :param data: Data to calculate over
        :param num_periods: The number of periods for each rolling calculation
        :return: Returns a pandas DataFrame of calculated values: first column is "aroondown", second column is
        "aroonup"
        :rtype: pandas.DataFrame
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        return ta.AROON(data, timeperiod=num_periods)

    @Alias('Aroonosc', 'AROONOSC', 'AroonOSC', 'AroonOsc')
    def aroon_oscillator(self, data, num_periods: int = 14) -> pd.Series:
        """
        The Aroon Oscillator is calculated by subtracting the Aroon Down from the Aroon Up. The resultant number will
        oscillate between 100 and -100. The Aroon Oscillator will be high when the Aroon Up is high and the Aroon Down
        is low, indicating a strong upward trend. The Aroon Oscillator will be low when the Aroon Down is high and
        the Aroon Up is low, indicating a strong downward trend. When the Up and Down are approximately equal,
        the Aroon Oscillator will hover around zero, indicating a weak trend or consolidation. See the Aroon
        indicator for more information.

        *The Aroon Oscillator moves between -100 and 100. A high oscillator value is an indication of an uptrend
        while a low oscillator value is an indication of a downtrend. When Aroon Up remains high from consecutive new
        highs, the oscillator value will be high, following the uptrend. When  The Aroon Oscillator line can be included
        with or without the Aroon Up and Aroon Down when viewing a chart. Significant changes in the direction of the
        Aroon Oscillator can help to identify a new trend.

        :param data: Data to calculate over
        :param num_periods: The number of periods for each rolling calculation
        :return: Returns a pandas Series of calculated values
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        return ta.AROONOSC(data, timeperiod=num_periods)

    @Alias('mfi', 'MFI')
    def money_flow_index(self, data, num_periods: int = 14) -> pd.Series:
        """
        The Money Flow Index calculates the ratio of money flowing into and out of a security. To interpret the Money
        Flow Index, look for divergence with price to signal reversals. Money Flow Index values range from 0 to 100.
        Values above 80/below 20 indicate market tops/bottoms.

        *One of the primary ways to use the Money Flow Index is when there is a divergence. A divergence is when the
        oscillator is moving in the opposite direction of price. This is a signal of a potential reversal in the
        prevailing price trend. For example, a very high Money Flow Index that begins to fall below a reading of 80
        while the underlying security continues to climb is a price reversal signal to the downside. Conversely,
        a very low MFI reading that climbs above a reading of 20 while the underlying security continues to sell off
        is a price reversal signal to the upside. Other moves out of overbought or oversold territory can also be
        useful. For example, when an asset is in an uptrend, a drop below 20 (or even 30) and then a rally back above
        it could indicate a pullback is over and the price uptrend is resuming. The same goes for a downtrend. A
        short-term rally could push the MFI up to 70 or 80, but when it drops back below that could be the time to
        enter a short trade in preparation for another drop.


        :param data: Data to calculate over
        :param num_periods: The number of periods for each rolling calculation
        :return: Returns a pandas Series of calculated values
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        return ta.MFI(data, timeperiod=num_periods)

    @Alias('TRIX', '1ROC_TEMA', '1ROC_T3')
    def trix(self, data, num_periods: int = 10, series_type: str = 'close') -> pd.Series:
        """
        The TRIX indicator calculates the rate of change of a triple exponential moving average. The values oscillate
        around zero. Buy/sell signals are generated when the TRIX crosses above/below zero. A (typically) 9 period
        exponential moving average of the TRIX can be used as a signal line. A buy/sell signals are generated when
        the TRIX crosses above/below the signal line and is also above/below zero.

        *As a powerful oscillator indicator, TRIX can be used to identify oversold and overbought markets, and it can
        also be used as a momentum indicator. Like many oscillators, TRIX oscillates around a zero line. When it is
        used as an oscillator, a positive value indicates an overbought market while a negative value indicates an
        oversold market. When TRIX is used as a momentum indicator, a positive value suggests momentum is increasing
        while a negative value suggests momentum is decreasing. Many analysts believe that when the TRIX crosses
        above the zero line, it gives a buy signal, and when it closes below the zero line, it gives a sell signal.
        Also, any divergence between price and TRIX can indicate significant turning points in the market.

        :param data: Data to calculate over
        :param num_periods: The number of periods for each rolling calculation
        :param series_type: The pricing data to calculate over
        :return: Returns a pandas Series of calculated values
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        series_type = check_series_type(series_type)
        return ta.TRIX(data, timeperiod=num_periods, price=series_type)

    @Alias('ultosc', 'ULTOSC')
    def ultimate_oscillator(self, data, num_periods1: int = 7, num_periods2: int = 14,
                            num_periods3: int = 28) -> pd.Series:
        """
        The Ultimate Oscillator is the weighted sum of three oscillators of different time periods. The typical time
        periods are 7, 14 and 28. The values of the Ultimate Oscillator range from zero to 100. Values over 70
        indicate overbought conditions, and values under 30 indicate oversold conditions. Also look for
        agreement/divergence with the price to confirm a trend or signal the end of a trend.

        *False divergences are common in oscillators that only use one time frame, because when the price surges the
        oscillator surges. Even if the price continues to rise the oscillator tends to fall forming a divergence even
        though the price may still be trending strongly. In order for the indicator to generate a buy signal,
        Williams recommended a three-step approach. 1, a bullish divergence must form. This is when the price makes a
        lower low but the indicator is at a higher low. 2, the first low in the divergence (the lower one) must have
        been below 30. This means the divergence started from oversold territory and is more likely to result in an
        upside price reversal. 3, the Ultimate oscillator must rise above the divergence high. The divergence high is
        the high point between the two lows of the divergence. Williams created the same three-step method for sell
        signals. 1, a bearish divergence must form. This is when the price makes a higher high but the indicator is
        at a lower high. 2, the first high in the divergence (the higher one) must be above 70. This means the
        divergence started from overbought territory and is more likely to result in a downside price reversal. 3,
        the Ultimate oscillator must drop below the divergence low. The divergence low is the low point between
        the two highs of the divergence.

        :param data: Data to calculate over
        :param num_periods1: The number of periods for each of the first rolling calculations
        :param num_periods2: The number of periods for each of the second rolling calculations
        :param num_periods3: The number of periods for each of the third rolling calculations
        :return: Returns a pandas Series of calculated values
        :rtype: pandas.Series
        """
        assert num_periods1 > 0, 'num_periods1 must be greater than 0'
        assert num_periods2 > num_periods1, 'num_periods2 must be greater than num_periods1'
        assert num_periods3 > num_periods2, 'num_periods3 must be greater than num_periods2'
        return ta.ULTOSC(data, timeperiod1=num_periods1, timeperiod2=num_periods2,
                         timeperiod3=num_periods3)

    @Alias('dx', 'DX')
    def directional_movement_index(self, data, num_periods: int = 14) -> pd.Series:
        """
        The DX is usually smoothed with a moving average (i.e. the ADX). The values range from 0 to 100, but rarely
        get above 60. To interpret the DX, consider a high number to be a strong trend, and a low number, a weak trend.

        *Crossovers are the main trade signals. A long trade is taken when the +DI crosses above -DI and uptrend
        could be underway. A sell signal occurs when the -DI drops below -DI. A short trade is initiated when -DI
        drops below +DI because a downtrend could be underway. While this method may produce some good signals,
        it will also produce some bad ones since a trend may not necessarily develop after entry. The indicator can
        also be used as a trend or trade confirmation tool. If the +DI is well above -DI, the trend has strength to
        the upside and this would help confirm current long trades or new long trade signals based on other entry
        methods. If -DI is well above +DI this confirms the strong downtrend or short positions.

        :param data: Data to calculate over
        :param num_periods: The number of periods for each rolling calculation
        :return: Returns a pandas Series of calculated values
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        return ta.DX(data, timeperiod=num_periods)

    @Alias('minus_di', 'MINUS_DI')
    def minus_directional_indicator(self, data, num_periods: int = 14) -> pd.Series:
        """
        The +DI is the percentage of the true range that is up. The -DI is the percentage of the true range that is
        down. A buy signal is generated when the +DI crosses up over the -DI. A sell signal is generated when the -DI
        crosses up over the +DI. You should wait to enter a trade until the extreme point is reached. That is,
        you should wait to enter a long trade until the price reaches the high of the bar on which the +DI crossed
        over the -DI, and wait to enter a short trade until the price reaches the low of the bar on which the -DI
        crossed over the +DI.

        :param data: Data to calculate over
        :param num_periods: The number of periods for each rolling calculation
        :return: Returns a pandas Series of calculated values
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        return ta.MINUS_DI(data, timeperiod=num_periods)

    @Alias('plus_di', 'PLUS_DI')
    def plus_directional_indicator(self, data, num_periods: int = 14) -> pd.Series:
        """
        The +DI is the percentage of the true range that is up. The -DI is the percentage of the true range
        that is down. A buy signal is generated when the +DI crosses up over the -DI. A sell signal is generated when
        the -DI crosses up over the +DI. You should wait to enter a trade until the extreme point is reached. That is,
        you should wait to enter a long trade until the price reaches the high of the bar on which the +DI
        crossed over the -DI, and wait to enter a short trade until the price reaches the low of the bar on which the
        -DI crossed over the +DI.

        :param data: Data to calculate over
        :param num_periods: The number of periods for each rolling calculation
        :return: Returns a pandas Series of calculated values
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        return ta.PLUS_DI(data, timeperiod=num_periods)

    @Alias('minus_dm', 'MINUS_DM')
    def minus_directional_movement(self, data, num_periods: int = 14) -> pd.Series:
        """
        The +DI is the percentage of the true range that is up. The -DI is the percentage of the true range
        that is down. A buy signal is generated when the +DI crosses up over the -DI. A sell signal is generated when
        the -DI crosses up over the +DI. You should wait to enter a trade until the extreme point is reached. That
        is, you should wait to enter a long trade until the price reaches the high of the bar on which the +DI crossed
        over the -DI, and wait to enter a short trade until the price reaches the low of the bar on which the -DI
        crossed over the +DI.

        :param data: Data to calculate over
        :param num_periods: The number of periods for each rolling calculation
        :return: Returns a pandas Series of calculated values
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        return ta.MINUS_DM(data, timeperiod=num_periods)

    @Alias('plus_dm', 'PLUS_DM')
    def plus_directional_movement(self, data, num_periods: int = 14) -> pd.Series:
        """
        The +DI is the percentage of the true range that is up. The -DI is the percentage of the true range
        that is down. A buy signal is generated when the +DI crosses up over the -DI. A sell signal is generated when
        the -DI crosses up over the +DI. You should wait to enter a trade until the extreme point is reached. That
        is, you should wait to enter a long trade until the price reaches the high of the bar on which the +DI crossed
        over the -DI, and wait to enter a short trade until the price reaches the low of the bar on which the -DI
        crossed over the +DI.

        :param data: Data to calculate over
        :param num_periods: The number of periods for each rolling calculation
        :return: Returns a pandas Series of calculated values
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        return ta.PLUS_DM(data, timeperiod=num_periods)

    @Alias('bbands', 'BBANDS', 'Bollinger_bands')
    def bollinger_bands(self, data, num_periods: int = 5, dev_up: int = 2, dev_dw: int = 2,
                        matype: int = 0) -> pd.DataFrame:
        """
        Bollinger Bands consist of three lines. The middle band is a simple moving average (generally 20 periods) of
        the typical price (TP). The upper and lower bands are F standard deviations (generally 2) above and below the
        middle band. The bands widen and narrow when the volatility of the price is higher or lower, respectively.

        Bollinger Bands do not, in themselves, generate buy or sell signals; they are an indicator of overbought or
        oversold conditions. When the price is near the upper or lower band it indicates that a reversal may be
        imminent. The middle band becomes a support or resistance level. The upper and lower bands can also be
        interpreted as price targets. When the price bounces off of the lower band and crosses the middle band,
        then the upper band becomes the price target.

        *The squeeze is the central concept of Bollinger Bands®. When the bands come close together, constricting the
        moving average, it is called a squeeze. A squeeze signals a period of low volatility and is considered by
        traders to be a potential sign of future increased volatility and possible trading opportunities. Conversely,
        the wider apart the bands move, the more likely the chance of a decrease in volatility and the greater the
        possibility of exiting a trade. However, these conditions are not trading signals. The bands give no
        indication when the change may take place or which direction price could move. Approximately 90% of price
        action occurs between the two bands. Any breakout above or below the bands is a major event. The breakout is
        not a trading signal. The mistake most people make is believing that that price hitting or exceeding one of
        the bands is a signal to buy or sell. Breakouts provide no clue as to the direction and extent of future
        price movement.

        :param data: Data to calculate over
        :param num_periods: he number of periods for each rolling calculation
        :param dev_up: The standard deviation multiplier for the upper band
        :param dev_dw: The standard deviation multiplier for the lower band
        :param matype: The arbitrary rolling average to use to calculate the middle band
        :return: Returns a pandas DataFrame: first column is "upperband", second column is "middleband", third column
        is "lowerband"
        :rtype: pandas.DataFrame
        """
        assert dev_up > 0, 'dev_up must be greater than 0'
        assert dev_dw > 0, 'dev_dw must be greater than 0'
        assert num_periods > 0, 'num_periods must be greater than 0'
        matype = check_matype(matype, 'matype')
        return ta.BBANDS(data, timeperiod=num_periods, nbdevup=dev_up, nbdevdn=dev_dw, matype=matype)

    @Alias('MIDPOINT')
    def midpoint(self, data, num_periods: int = 14, series_type: str = 'close') -> pd.Series:
        """
        The rolling Midpoint of the specified pricing data.

        :param data: Data to calculate over
        :param num_periods: The number of periods for each rolling calculation
        :param series_type: The pricing data to calculate over
        :return: Returns a pandas Series of calculated values
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        series_type = check_series_type(series_type)
        return ta.MIDPOINT(data, timeperiod=num_periods, price=series_type)

    @Alias('MIDPRICE')
    def midprice(self, data, num_periods: int = 14) -> pd.Series:
        """
        The rolling Midprice of the specified pricing data calculated by summing the highest high and lowest low and
        then dividing by two.

        :param data: Data to calculate over
        :param num_periods: The number of periods for each rolling calculation
        :return: Returns a pandas Series of calculated values
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        return ta.MIDPRICE(data, timeperiod=num_periods)

    @Alias('sar', 'SAR')
    def parabolic_sar(self, data, acceleration: float = 0.01, maximum: float = 0.2) -> pd.Series:
        """
        The Parabolic SAR calculates a trailing stop. Simply exit when the price crosses the SAR. The SAR assumes
        that you are always in the market, and calculates the Stop And Reverse point when you would close a long
        position and open a short position or vice versa.

        *Typically place stop loss orders at the Parabolic SAR values

        :param data: Data to calculate over
        :param acceleration: A scaling factor to capture momentum
        :param maximum: The lowest or highest point depending on the trend
        :return: Returns a pandas Series of calculated values
        :rtype: pandas.Series
        """
        assert acceleration >= 0, 'acceleration must be greater than or equal to 0'
        assert maximum >= 0, 'maximum must be greater than or equal to 0'
        return ta.SAR(data, acceleration=acceleration, maximum=maximum)

    @Alias('trange', 'TRANGE', 'TRange')
    def true_range(self, data) -> pd.Series:
        """
        The True Range function is used in the calculation of many indicators, most notably, the Welles Wilder DX. It
        is a base calculation that is used to determine the normal trading range of a stock or commodity.

        :param data: Data to calculate over
        :return: Returns a pandas Series of calculated values
        :rtype: pandas.Series
        """
        return ta.TRANGE(data)

    @Alias('atr', 'ATR', 'AvgTRANGE', 'AvgTRange')
    def average_true_range(self, data, num_periods: int = 14) -> pd.Series:
        """
        The ATR is a Welles Wilder style moving average of the True Range. The ATR is a measure of volatility. High
        ATR values indicate high volatility, and low values indicate low volatility, often seen when the price is flat.

        The ATR is a component of the Welles Wilder Directional Movement indicators (+/-DI, DX, ADX and ADXR).

        *Measures the volatility of an asset, used with the Chandelier Exit strategy

        :param data: Data to calculate over
        :param num_periods: The number of periods for each rolling calculation
        :return: Returns a pandas Series of calculated values
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        return ta.ATR(data, timeperiod=num_periods)

    @Alias('natr', 'NATR')
    def normalized_average_true_range(self, data, num_periods: int = 14) -> pd.Series:
        """
        The ATR is a Welles Wilder style moving average of the True Range. The ATR is a measure of volatility. High
        ATR values indicate high volatility, and low values indicate low volatility, often seen when the price is flat.

        The ATR is a component of the Welles Wilder Directional Movement indicators (+/-DI, DX, ADX and ADXR).

        :param data: Data to calculate over
        :param num_periods: The number of periods for each rolling calculation
        :return: Returns a pandas Series of normalized calculated values
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        return ta.NATR(data, timeperiod=num_periods)

    @Alias('ad', 'AD', 'Chaikin_AD_Line', 'Chaikin_AD_line', 'chaikin_ad_line')
    def chaikin_ad_line_values(self, data) -> pd.Series:
        """
        The Accumulation/Distribution Line is similar to the On Balance Volume (OBV), which sums the volume times
        +1/-1 based on whether the close is higher than the previous close. The Accumulation/Distribution indicator,
        however multiplies the volume by the close location value (CLV). The CLV is based on the movement of the
        issue within a single bar and can be +1, -1 or zero.

        *The Accumulation/Distribution Line is interpreted by looking for a divergence in the direction of the
        indicator relative to price. If the Accumulation/Distribution Line is trending upward it indicates that the
        price may follow. Also, if the Accumulation/Distribution Line becomes flat while the price is still rising (
        or falling) then it signals an impending flattening of the price.

        :param data: Data to calculate over
        :return: Returns a pandas Series of calculated values
        :rtype: pandas.Series
        """
        return ta.AD(data)

    @Alias('adosc', 'ADOSC', 'Chaikin_AD_Oscillator')
    def chaikin_ad_oscillator(self, data, fast: int = 3, slow: int = 10) -> pd.Series:
        """
        The Chaikin Oscillator (AKA Chaikin A/D Oscillator) is essentially a momentum of the
        Accumulation/Distribution Line. It is calculated by subtracting a 10 period exponential moving average of the
        A/D Line from a 3 period exponential moving average of the A/D Line. When the Chaikin Oscillator crosses
        above zero, it indicates a buy signal, and when it crosses below zero it indicates a sell signal. Also look
        for price divergence to indicate bullish or bearish conditions.

        *The Accumulation/Distribution Line is interpreted by looking for a divergence in the direction of the
        indicator relative to price. If the Accumulation/Distribution Line is trending upward it indicates that the
        price may follow. Also, if the Accumulation/Distribution Line becomes flat while the price is still rising (
        or falling) then it signals an impending flattening of the price.

        :param data: Data to calculate over
        :param fast: The time period to use for the fast exponential moving average
        :param slow: The time period to use for the slow exponential moving average
        :return: Returns a pandas Series of calculated values
        :rtype: pandas.Series
        """
        assert fast > 0, 'fast must be greater than 0'
        assert slow > fast, 'slow must be greater than fast'
        return ta.ADOSC(data, fastperiod=fast, slowperiod=slow)

    @Alias('obv', 'OBV', 'balance_volume')
    def on_balance_volume(self, data, num_periods: int = 5) -> pd.Series:
        """
        The On Balance Volume (OBV) is a cumulative total of the up and down volume. When the close is higher than the
        previous close, the volume is added to the running total, and when the close is lower than the previous close,
        the volume is subtracted from the running total.

        *The theory behind OBV is based on the distinction between smart money – namely, institutional investors –
        and less sophisticated retail investors. As mutual funds and pension funds begin to buy into an issue that
        retail investors are selling, volume may increase even as the price remains relatively level. Eventually,
        volume drives the price upward. At that point, larger investors begin to sell, and smaller investors begin
        buying. Despite being plotted on a price chart and measured numerically, the actual individual quantitative
        value of OBV is not relevant. The indicator itself is cumulative, while the time interval remains fixed by a
        dedicated starting point, meaning the real number value of OBV arbitrarily depends on the start date.
        Instead, traders and analysts look to the nature of OBV movements over time; the slope of the OBV line
        carries all of the weight of analysis. Analysts look to volume numbers on the OBV to track large,
        institutional investors. They treat divergences between volume and price as a synonym of the relationship
        between "smart money" and the disparate masses, hoping to showcase opportunities for buying against incorrect
        prevailing trends. For example, institutional money may drive up the price of an asset, then sell after other
        investors jump on the bandwagon. To interpret the OBV, look for the OBV to move with the price or precede
        price moves. If the price moves before the OBV, then it is a non-confirmed move. A series of rising peaks,
        or falling troughs, in the OBV indicates a strong trend. If the OBV is flat, then the market is not trending.

        :param data: Data to calculate over
        :param num_periods: The number of periods for each rolling calculation
        :return: Returns a pandas Series of calculated values
        :rtype: pandas.Series
        """
        assert num_periods > 0, 'num_periods must be greater than 0'
        return ta.OBV(data, timeperiod=num_periods)

    # Cycle Indicators

    @Alias('trendline', 'TRENDLINE', 'instantaneous')
    def hilbert_transform_instantaneous_trendline(self, data, series_type: str = 'close') -> pd.Series:
        series_type = check_series_type(series_type)
        return ta.HT_TRENDLINE(data, price=series_type)

    @Alias('sine', 'SINE', 'ht_sine', 'HT_SINE', 'sine_wave', 'SINE_WAVE')
    def hilbert_transform_sine_wave(self, data, series_type: str = 'close') -> pd.Series:
        series_type = check_series_type(series_type)
        return ta.HT_SINE(data, price=series_type)

    @Alias('trendmode', 'TRENDMODE', 'Trend_vs_Cycle')
    def hilbert_transform_trend_vs_cycle_mode(self, data, series_type: str = 'close') -> pd.Series:
        series_type = check_series_type(series_type)
        return ta.HTTRENDMODE(data, price=series_type)

    @Alias('dcperiod', 'DCPERIOD', 'dc_period', 'DC_PERIOD', 'Dominant_Cycle_Period')
    def hilbert_transform_dominant_cycle_period(self, data, series_type: str = 'close') -> pd.Series:
        series_type = check_series_type(series_type)
        return ta.HT_DCPERIOD(data, price=series_type)

    @Alias('dcphase', 'DCPHASE', 'dc_phase', 'DC_PHASE', 'Dominant_Cycle_Phase')
    def hilbert_transform_dominant_cycle_phase(self, data, series_type: str = 'close') -> pd.Series:
        series_type = check_series_type(series_type)
        return ta.HT_DCPHASE(data, price=series_type)

    @Alias('phasor', 'PHASOR', 'Phasor_Components')
    def hilbert_transform_phasor_components(self, data, series_type: str = 'close') -> pd.Series:
        series_type = check_series_type(series_type)
        return ta.HT_PHASOR(data, price=series_type)
