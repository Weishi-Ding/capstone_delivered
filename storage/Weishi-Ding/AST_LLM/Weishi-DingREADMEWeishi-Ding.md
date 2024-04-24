# README.md

## Project Title
**Advanced Q-Learning Models for Stock Trading Simulation**

## Introduction or Summary
This Python code repository is dedicated to developing advanced Q-learning models specifically tailored for stock trading simulations. The project integrates a series of Python files that collectively build a robust reinforcement learning environment. Starting with basic data manipulation and processing functionalities, the repository evolves to include neural network-based Q-learning algorithms, a dynamic stock trading environment simulation, and comprehensive agent training and portfolio management systems. Each component is designed to work in harmony, providing a realistic and effective framework for testing and refining stock trading strategies using reinforcement learning techniques.

## Technology Stack
- **Python**: The backbone programming language for all development, providing flexibility and a wide range of libraries.
- **NumPy**: Essential for high-performance numerical computing, especially array manipulations which are critical in data processing tasks.
- **PyTorch**: Utilized for building and training neural network models, leveraging GPU acceleration to enhance computational efficiency.
- **Pandas**: Employed for efficient data manipulation and preprocessing, crucial for handling structured stock trading data.
- **Tushare**: A specialized library for accessing real-time and historical data from the Chinese stock market, ensuring the models have access to accurate and current market data.
- **Matplotlib**: Used for visualizing data and model performance, aiding in the analysis and refinement of trading strategies.

## Engineering Highlights
- **Data Preprocessing Techniques**: Implementation of functions like `ndarray_to_tensor` ensures optimal data formatting for neural network processing.
- **Double Q-Learning Implementation**: Separation of current and target Q-values calculation demonstrates an advanced understanding of reinforcement learning nuances, reducing overestimations and improving model reliability.
- **Experience Replay and Buffer Management**: The `Experience_Buffer` class enhances training stability by managing how experiences are stored and utilized during model training.
- **Realistic Trading Environment**: The `Environment` class encapsulates complex trading dynamics, including stock selection and reward calculations, providing a realistic simulation for the reinforcement learning agent.
- **Portfolio Management Setup**: Initial preparations for the `Portfolio` class indicate a sophisticated approach to managing multiple stocks and strategies within the simulated environment.

## Features
- **Neural Network Architecture**: The `DQN` class defines the deep learning model structure, crucial for the Q-learning algorithm.
- **Experience Management**: The `Experience_Buffer` class supports efficient storage and sampling of agent experiences, a key feature for effective learning.
- **Dynamic Environment Simulation**: The `Environment` class accurately simulates a stock trading environment, complete with transaction handling and reward systems.
- **Comprehensive Data Preprocessing**: Functions like `normalize_column` and `add_avg` prepare the trading data by normalizing and enriching it with calculated features.
- **Integrated Training Workflow**: The repository structures agent training, backtesting, and portfolio management into a cohesive workflow, facilitating the development and evaluation of trading strategies.

## Usage
To get started with this project, follow these steps:

1. **Setting Up the Environment**:
   ```python
   from environment import Environment
   train_env = Environment(train_data)
   val_env = Environment(val_data)
   ```

2. **Initializing and Training the Agent**:
   ```python
   from agent import Agent
   agent = Agent(model=DQN())
   agent.train(train_env, val_env, num_episodes=50, batch_size=32, epsilon=0.1, gamma=0.99, lr=0.01, device='cuda')
   ```

3. **Running Backtests**:
   ```python
   from backtest import Backtest
   test_env = Environment(test_data)
   backtest = Backtest(agent, test_env)
   backtest.execute()
   ```

4. **Portfolio Management**:
   ```python
   from portfolio import Portfolio
   portfolio = Portfolio(stock_list)
   portfolio.initialize()
   portfolio.run()
   portfolio.evaluate()
   ```

These instructions provide a basic setup and operation guide. For detailed usage and additional functionalities, refer to the specific module documentation within the repository.

## Closing Remark
This README offers a concise overview of the project's capabilities and setup. For more detailed information on specific functionalities and extended examples, please refer to the individual file documentation within the repository. This will enable users to fully leverage the advanced features and capabilities of the Q-learning models for stock trading simulations.