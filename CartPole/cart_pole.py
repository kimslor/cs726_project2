import gym
import numpy as np
import rl_agent
import rl_model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

def initialize_model():
    # Create environment and get parameters
    env = gym.make('CartPole-v1')
    actions = env.action_space.n

    # Create model and agent for training
    model = rl_model.build_model(
        state_shape=env.observation_space.shape,
        actions=actions)
    dqn = rl_agent.build_agent(
        model=model,
        actions=actions)

    return env, model, dqn

def train_model() -> None:
    env, model, dqn = initialize_model()

    # Start training
    dqn.fit(
        env=env,
        nb_steps=10000,
        visualize=False,
        verbose=2)

    # Save model
    dqn.save_weights('SavedWeights/10k/dqn_weights.h5f')

    # Close environment
    del model, dqn
    env.close()

def test_model() -> None:
    env, model, dqn = initialize_model()

    # Load weights
    dqn.load_weights('SavedWeights/100k/dqn_weights.h5f')

    # Test model
    scores = dqn.test(
        env=env,
        visualize=False,
        nb_episodes=10,
        verbose=2)
    print(np.mean(scores.history['episode_reward']))

    # Close environment
    del model, dqn
    env.close()

if __name__ == "__main__":
    #train_model()
    test_model()
