from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from tensorflow.keras.models import Sequential


def build_agent(
    model: Sequential,
    actions: int
) -> DQNAgent:
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr='eps',
        value_max=1.,
        value_min=.1,
        value_test=.05,
        nb_steps=10000)
    memory = SequentialMemory(
        limit=50000,
        window_length=3)
    dqn = DQNAgent(
        model=model,
        memory=memory,
        policy=policy,
        enable_dueling_network=True,
        dueling_type='avg',
        nb_actions=actions,
        nb_steps_warmup=100)
    return dqn
