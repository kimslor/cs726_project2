import gym
import numpy as np
import rl_agent
import rl_model

from tensorflow.keras.optimizers import Adam


def initialize_model(visualize):
    # Create environment and get parameters
    if (visualize):
        env = gym.make('ALE/SpaceInvaders-v5', render_mode='human')
    else:
        env = gym.make('ALE/SpaceInvaders-v5')
    height, width, channels = env.observation_space.shape
    actions = env.action_space.n

    # Create model and agent for training
    model = rl_model.build_model(
        height=height,
        width=width,
        channels=channels,
        actions=actions)
    dqn = rl_agent.build_agent(
        model=model,
        actions=actions)
    dqn.compile(Adam())

    model.summary()

    return env, model, dqn

def train_model() -> None:
    env, model, dqn = initialize_model(False)

    # Start training
    dqn.fit(
        env=env,
        nb_steps=100000,
        verbose=2)

    # Save model
    dqn.save_weights('SavedWeights/100k/dqn_weights.h5f')

    # Close environment
    del model, dqn
    env.close()

def test_model() -> None:
    env, model, dqn = initialize_model(False)

    # Load a trained model
    dqn.load_weights('SavedWeights/Old_Weights/100k/dqn_weights.h5f')

    # Test model
    scores = dqn.test(
        env=env,
        nb_episodes=10,
        visualize=False)
    print(np.mean(scores.history['episode_reward']))

    # Close environment
    del model, dqn
    env.close()

if __name__ == "__main__":
    #train_model()
    test_model()
