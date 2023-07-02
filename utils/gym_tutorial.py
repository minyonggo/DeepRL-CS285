import random
import gym

from matplotlib import animation
import matplotlib.pyplot as plt



def save_frames_as_gif(frames, path='./', filename='gym_tutorial.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)





env = gym.make("CartPole-v1")

episodes = 5
frames = []

discrete = isinstance(env.action_space, gym.spaces.Discrete)
ob_dim = env.observation_space.shape[0]
ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

print(f"States: {ob_dim}, Actions: {ac_dim}, Discrete: {discrete}")
print(f"Episode Steps: {env.spec.max_episode_steps}")

for episode in range(episodes):
    state = env.reset()
    done = False
    score = 0

    while not done:
        frames.append(env.render(mode="rgb_array"))

        # action = random.choice([0, 1])
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action=action)
        score += reward
        

    print(f"Episode {episode}, Score: {score}")

env.close()
save_frames_as_gif(frames)
