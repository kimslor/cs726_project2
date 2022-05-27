from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from tensorflow.keras.models import Sequential

NB_STEPS = 50000

def build_agent(
    model: Sequential,
    actions: int
) -> DQNAgent:
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr='eps',
        value_max=1.,
        value_min=.1,
        value_test=.2,
        nb_steps=NB_STEPS)
    memory = SequentialMemory(
        limit=1000,
        window_length=3)
    dqn = DQNAgent(
        model=model,
        memory=memory,
        policy=policy,
        enable_dueling_network=True,
        dueling_type='avg',
        nb_actions=actions,
        nb_steps_warmup=1000)
    return dqn
