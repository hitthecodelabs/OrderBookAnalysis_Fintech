# OrderBookAnalysis_Fintech

## Overview

This Python script embodies a sophisticated algorithmic trading bot meticulously crafted for the volatile cryptocurrency futures market. It operates on the principle of deep order book analysis, extracting granular market microstructure data from multiple exchanges, and leverages pre-trained machine learning models to anticipate short-term price movements.  The bot is designed to execute automated trading strategies on the Bybit exchange, managing both long and short positions with integrated risk management features.

This project is a testament to continuous development and refinement, reflecting an ongoing pursuit of optimal trading performance through advanced data analysis and machine learning techniques. It is not a plug-and-play solution but rather a complex system requiring careful configuration, understanding, and continuous monitoring.

**Warning:** Trading cryptocurrencies, especially with leverage as implemented in this bot, carries substantial financial risk. This bot is provided as-is, for informational and educational purposes only. Use it at your own risk. The developers assume no liability for any financial losses incurred while using this software. **Thorough backtesting, paper trading, and a deep understanding of the code are absolutely mandatory before deploying with real capital.**

## Key Features

*   **Multi-Exchange Order Book Data Ingestion:**  Capable of retrieving real-time Level 2 order book data from major cryptocurrency exchanges including Binance, OKX, Bitstamp, Coinbase Pro, and Bybit. This allows for a comprehensive view of market liquidity and depth.
*   **Advanced Order Book Feature Engineering:** Implements a suite of custom functions to process raw order book data into meaningful features. These features capture:
    *   **Volume Distribution:**  Volume at various depth levels (5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 250 levels deep on both bid and ask sides).
    *   **Volume Ratios:** Relative bid and ask volume at different depths, normalized to represent market pressure and order imbalance.
    *   **Price Variation at Depth:** Percentage price differences between the top level and deeper levels of the order book, indicating potential price elasticity and order book slope.
    *   **Spread Calculation:** Real-time bid-ask spread, a crucial indicator of market liquidity and trading costs.
*   **Machine Learning Driven Prediction (Pre-trained Models):** Integrates with pre-trained Keras/TensorFlow deep learning models. These models are designed to predict short-term price directionality based on the engineered order book features.  *(Note: Model training and architecture are not included in this script and are assumed to be performed separately).*
*   **Automated Bybit Futures Trading:** Executes trading signals directly on the Bybit USDT Perpetual futures exchange using the `pybit` API. Supports market orders for both entry and exit.
*   **Dynamic Stop-Loss Implementation:**  Employs a dynamic stop-loss mechanism that adjusts the stop-loss level upwards as the trade moves into profit. This aims to lock in gains while limiting potential losses. The stop-loss is calculated based on a percentage of the maximum favorable excursion (profit peak) of the trade.
*   **Telegram and Twitter Notifications:** Provides comprehensive real-time notifications via:
    *   **Telegram:**  Detailed trade execution reports (openings and closings), profit/loss summaries, error alerts, and session status updates are sent to dedicated Telegram channels. Multiple bot tokens are used for load balancing and redundancy.
    *   **Twitter:**  Automated tweets are generated for trade openings and closings, including entry/exit prices, coin, trade direction (long/short), and relevant hashtags. Replies are posted to closing tweets to thread trade information.
*   **Robust Error Handling and Session Management:** Implements `try-except` blocks to catch and handle common API errors, network connectivity issues, and other exceptions.  Includes mechanisms to gracefully close open positions and report critical errors via Telegram if the bot encounters fatal errors or session disruptions.
*   **Data Logging for Analysis and Model Improvement:**  Logs raw order book data and engineered feature vectors to CSV files. This historical data is intended for:
    *   **Performance Analysis:** Evaluating the bot's trading performance over time.
    *   **Feature Engineering Refinement:** Analyzing feature importance and identifying potential improvements.
    *   **Model Retraining:**  Providing updated data for retraining machine learning models to adapt to changing market dynamics.
*   **Modular Design with `vannamei` Class:**  The core trading logic is encapsulated within the `vannamei` class, promoting code organization and maintainability. Two distinct methods within this class (`__white__` and `__lotus__`) represent different operational modes or developmental stages of the bot.
*   **Configurable Parameters:**  Key parameters such as leverage, stop-loss percentages, trading amounts, and API keys are intended to be configured externally via a `config.py` file, enhancing flexibility and security.

## Functionality Deep Dive

The script's operation can be broken down into the following key stages:

1.  **Initialization (`vannamei` Class, `__white__` and `__lotus__` methods):**
    *   **API Client Setup:** Initializes API clients for Twitter (`tweepy`), ApolloX (`ApolloxC`), and Bybit (`pybit.usdt_perpetual.HTTP`).
    *   **Leverage Configuration:** Sets the trading leverage for Bybit futures contracts (currently hardcoded to 4x in `__white__` and `__lotus__`, but intended to be configurable).
    *   **Coin Information Loading:** Loads coin-specific data (e.g., Telegram bot mappings, Twitter hashtags) from `coins_data.json`.
    *   **File System Setup:** Creates necessary directories (`coin{coin}`, `vectors`) for storing data logs.
    *   **Data File Creation:**  Initializes CSV files for storing raw order book data (`nFile1`) and feature vectors (`name_v`).

2.  **Real-time Order Book Data Acquisition (`order_book_file`, `cheese*` functions):**
    *   **Exchange API Calls:**  The `order_book_file` function (and its variant `order_book_filee`) is responsible for making API calls to cryptocurrency exchanges to retrieve Level 2 order book snapshots for a specified trading pair (e.g., `BTCUSDT`). The script supports Binance Futures (USD and COIN-margined), OKX Swaps, Bitstamp, Coinbase Pro, and Bybit.
    *   **Exchange-Specific Data Processing (`cheeseBinance`, `cheeseOKX_Kucoin`, `cheeseBitstamp`, `cheeseCOIN`, `cheeseBybit`):**  The `cheese*` functions are tailored to handle the specific data formats returned by each exchange's API. They perform initial parsing, data type conversion, and basic calculations like spread.
    *   **Data Normalization and Feature Calculation (`norm_order_book`, `perc_diff`, `pump_it_up`):**  These functions are central to feature engineering. `norm_order_book` normalizes prices around the mid-price and calculates percentage price changes at different order book depths. `perc_diff` calculates percentage price variations. `pump_it_up` computes volume-weighted prices, aggregating volume at different levels.
    *   **Data Storage:** Raw order book lines are written to temporal CSV files (`nFile1`) for potential historical analysis and to serve as input for feature extraction and model inference.

3.  **Feature Engineering and Vector Generation (`dict_data`, `ds_heikin_zz`, `m_G5X`, `inference_456X`, `vektor_line`, `get_vekctor`):**
    *   **Minute-wise Data Aggregation (`dict_data`):** The `dict_data` function aggregates incoming order book lines into minute-based blocks, preparing the data for time-series feature extraction.
    *   **Historical Data Resampling and Feature Calculation (`ds_heikin_zz`):**  `ds_heikin_zz` reads historical order book data (from `nFile1`), resamples it to 1-minute intervals, and calculates volume and volume ratio features (`dvol`, `dvor`) using `stts`. It also fetches 1-minute Kline data (OHLC) from ApolloX or Binance to incorporate candlestick information.
    *   **Feature Vector Extraction (`m_G5X`, `m_G_5X`, `inference_456X`, `vektor_line`, `get_vekctor`):**  The `m_G5X` (and its variant `m_G_5X`) functions are crucial for generating the final feature vector. They take processed order book data (DataFrames from `ds_heikin_zz`) and calculate a fixed-length numerical vector that represents the market state. This vector includes:
        *   Volume ratios (`rb_r`, `ra_r`) at different depths.
        *   Volume variations (`vb_r`, `va_r`) at different depths.
        *   Price variations (`diff_ha`).
    *   **Vector Storage:**  Feature vectors are saved to CSV files (`name_v`) for potential model retraining or offline analysis.

4.  **Machine Learning Inference (Commented Out in Snippet - Intended Functionality):**
    *   **Model Loading (`get_utils`, `scaler_data`, `get_utilsss`, `#load_model`):**  The script is designed to load pre-trained Keras/TensorFlow models and associated scalers from files. The `get_utils` function (and related helpers) is responsible for locating and loading the correct model and scaler based on coin, time frame, and date. *(Note: Model training code is not included).*
    *   **Data Scaling (`StandardScaler`):**  Input feature vectors are scaled using a `StandardScaler` object (pre-fitted during model training) to ensure data normalization before feeding into the model.
    *   **Prediction Generation (`#loaded_model.predict`):**  The loaded machine learning model (`loaded_model`) is intended to be used to predict the probability of different market outcomes (e.g., long, short, neutral) based on the scaled feature vector.
    *   **Prediction Processing (Commented Out Logic):**  The commented-out code in `__white__` shows an intended logic for processing model predictions. This likely involves:
        *   Averaging predictions over a short time window (e.g., last `K_z` predictions).
        *   Filtering predictions based on probability thresholds (e.g., requiring probabilities above a certain level, like `prob_1 = 0.9825`).
        *   Using color predictions (likely classes output by the model - green, red, gray) to further refine trading signals.

5.  **Trading Logic and Order Execution (`vannamei` Class, `__white__` method - Commented Out):**
    *   **Signal Generation (Commented Out):**  The commented-out sections in `__white__` implement a trading strategy based on processed model predictions (`yesOnoD`, `yesOnoC`). The strategy appears to trigger trades based on:
        *   High average prediction probabilities (`yesOnoD == 1`).
        *   Consistent color predictions (e.g., all GREEN for long, all RED for short - `yesOnoC == 1` or `0`).
        *   Absence of an existing open position (`flag == ''`).
        *   Preventing repeated order openings in the same direction (`check != 'up'` or `check != 'down'`).
        *   Avoiding immediate re-entry after a stop-loss (`check_sl != 'g'` or `check_sl != 'r'`).
    *   **Order Placement (`open_long`, `open_short`, `open_order`, `sign_order`):** When trading conditions are met, the `open_long` or `open_short` functions are called. These functions:
        *   Calculate the trade quantity based on a predefined budget (`current_p`) and leverage (`to_speculate`).
        *   Determine the appropriate order price (bid for long, ask for short).
        *   Calculate liquidation price and stop-loss level (`calc_liq_price`).
        *   Place market orders on Bybit using `session.place_active_order` (via `sign_order`).
        *   Record trade details (entry price, quantity, leverage, side).
    *   **Position Closing (`close_long`, `close_short`, `cerrar_orden`):**  Positions are intended to be closed under the following conditions:
        *   **Opposite Trading Signal:**  When the model predicts a reversal or neutral market condition (e.g., color predictions change from GREEN to RED or GRAY, or vice-versa).
        *   **Stop-Loss Trigger:**  (Implemented via dynamic stop-loss in `__white__` - commented out).
        *   **Error Condition:** (Handled in `try-except` blocks - closes positions in case of session errors).
        *   Position closing is executed using `session.close_position` (via `cerrar_orden`).

6.  **Risk Management (Dynamic Stop-Loss - Commented Out):**
    *   **Stop-Loss Calculation (`calc_liq_price`):**  Initial stop-loss levels are calculated when opening a position based on a percentage of the entry price, leverage, and trade direction.
    *   **Dynamic Stop-Loss Adjustment (Commented Out Logic in `__white__`):**  The commented-out stop-loss monitor block in `__white__` is designed to dynamically adjust the stop-loss upward as the trade becomes profitable. This strategy aims to:
        *   **Limit Downside Risk:**  The initial stop-loss protects against immediate adverse price movements.
        *   **Lock in Profits:**  The dynamic adjustment moves the stop-loss to breakeven or into profit as the trade progresses favorably, securing gains.
        *   **Stop-Loss Update (`update_sl`):**  The `update_sl` function is used to send API requests to Bybit to modify the stop-loss order for an active position.

7.  **Notifications and Reporting (`send_activity`, `send_activity2`, Twitter functions):**
    *   **Telegram Notifications:** The `send_activity` and `send_activity2` functions are used extensively to provide real-time updates via Telegram. These notifications cover:
        *   Trade Openings and Closings: Including entry/exit prices, coin, quantity, profit/loss, and leverage.
        *   Error Reports:  Reporting API errors, session disruptions, and other critical issues.
        *   Session Status:  Indicating when the bot session starts or terminates.
    *   **Twitter Integration (Optional):**  The script includes functions (`post_tweet`, `post_reply`) for posting automated updates to Twitter. These tweets are intended to:
        *   Publicly announce trade openings and closings.
        *   Potentially engage with a trading community or audience (using relevant hashtags like `#algotrade`, `#cryptotrading`, `#stewiebot`).

8.  **File Management and Data Persistence:**
    *   **Order Book Data Logging:** Raw order book data is continuously written to temporal CSV files (`nFile1`).
    *   **Feature Vector Logging:** Engineered feature vectors are saved to CSV files (`name_v`).
    *   **File Rotation (Temporal Data):**  The script implements a file rotation mechanism (in `__white__`) to manage the size of the temporal order book data file (`nFile1`). It periodically creates copies of the file and truncates the main file to keep only the most recent data.
    *   **(Commented Out) Dropbox Integration:**  Commented-out code suggests potential integration with Dropbox for data backup or model storage, although this is not currently active in the provided script.

## `vannamei` Class: `__white__` vs. `__lotus__`

The `vannamei` class contains two distinct methods, `__white__` and `__lotus__`, which represent different operational modes or stages in the bot's development:

*   **`__white__(params)`:**  This method appears to be the more feature-complete and intended mode for live trading or comprehensive backtesting. It includes:
    *   Full trading logic (though commented out in the snippet) with signal generation, order execution, and position management.
    *   Dynamic stop-loss monitoring and adjustment (commented out).
    *   File rotation and temporal data management.
    *   Intended for a more autonomous, end-to-end trading operation.

*   **`__lotus__(params)`:**  This method seems to be a more focused or experimental mode, primarily centered around:
    *   Real-time order book data acquisition and processing.
    *   Feature extraction and vector generation.
    *   Data logging of feature vectors.
    *   Lacks the explicit trading logic, stop-loss management, and complete automation found in `__white__`.
    *   Potentially intended for data collection, feature engineering validation, or as a data preprocessing step for a separate trading system or model.

The presence of these two methods suggests a phased development approach, where `__lotus__` might represent an earlier stage focused on data and feature engineering, while `__white__` represents a more advanced stage aiming for full automated trading functionality.
