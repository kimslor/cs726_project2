from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def build_agent(
    model: Sequential,
    actions: int
) -> DQNAgent:
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(
        limit=50000,
        window_length=1)
    dqn = DQNAgent(
        model=model,
        memory=memory,
        policy=policy,
        target_model_update=0.01,
        nb_actions=actions,
        nb_steps_warmup=10)
    dqn.compile(Adam())
    return dqn
