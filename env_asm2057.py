import pygame
import numpy as np
import gymnasium as gym
import random  # Import random module for random action policy

# Define constants for the screen width and height
SCREEN_WIDTH = 700
SCREEN_HEIGHT = 700

# Define constants for the grid size
GRID_SIZE = 7
CELL_SIZE = SCREEN_WIDTH // GRID_SIZE

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Define positions for the "goal" state and "hell" states
GOAL_STATE = (6, 6)
HELL_STATES = [(2, 1), (1, 4), (5, 3)]

# Define the quit key
QUIT_KEY = pygame.K_q  # Change this to any key you want to use for quitting

# Step 1: Define your own custom environment
class CustomEnv(gym.Env):
    """
    Custom environment for a 7x7 grid world using the Gymnasium interface.
    
    """
    def __init__(self, grid_size=7) -> None:
        """
        Initializes the custom environment.
        
        Args:
            grid_size (int): Size of the grid. Default is 7.
        """
        super(CustomEnv, self).__init__()
        self.grid_size = grid_size
        self.cell_size = CELL_SIZE
        self.state = None
        self.reward = 0
        self.info = {}
        self.goal = np.array(GOAL_STATE)
        self.done = False
        self.hell_states = [np.array(hell) for hell in HELL_STATES]

        # Action-space:
        self.action_space = gym.spaces.Discrete(4)
        
        # Observation space:
        self.observation_space = gym.spaces.Box(low=0, high=grid_size-1, shape=(2,), dtype=np.int32)

        # Initialize the window:
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("7x7 Grid")

        # Load images
        self.background_image = pygame.image.load("background.jpg").convert()
        self.background_image = pygame.transform.scale(self.background_image, (SCREEN_WIDTH, SCREEN_HEIGHT))

        self.player_image = pygame.image.load("player.png").convert_alpha()
        self.player_image = pygame.transform.scale(self.player_image, (CELL_SIZE, CELL_SIZE))

        self.goal_image = pygame.Surface((CELL_SIZE, CELL_SIZE))
        self.goal_image.fill(GREEN)

        self.hell_image = pygame.image.load("hell.png").convert_alpha()
        self.hell_image = pygame.transform.scale(self.hell_image, (CELL_SIZE, CELL_SIZE))

        self.goal_image = pygame.image.load("goal.png").convert_alpha()
        self.goal_image = pygame.transform.scale(self.goal_image, (CELL_SIZE, CELL_SIZE))

        self.goal_reached_image = pygame.image.load("goal_reached.png").convert_alpha()
        self.goal_reached_rect = self.goal_reached_image.get_rect()
        self.goal_reached_rect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)

        # Initialize font
        self.font = pygame.font.Font(None, 74)  # Use default font, size 74

    def reset(self):
        """
        Resets the environment to the initial state.
        
        Returns:
            tuple: A tuple containing the initial state and additional information.
        """
        self.state = np.array([0, 0])
        self.done = False
        self.reward = 0

        self.info["Distance to goal"] = np.sqrt(
            (self.state[0] - self.goal[0])**2 + 
            (self.state[1] - self.goal[1])**2
        )

        return self.state, self.info

    def step(self, action):
        """
        Executes a step in the environment based on the given action.
        
        Args:
            action (int): Action to be taken.
        
        Returns:
            tuple: A tuple containing the new state, reward, done flag, and additional information.
        """
        if action == 0 and self.state[0] > 0:  # Up
            self.state[0] -= 1
        elif action == 1 and self.state[0] < self.grid_size - 1:  # Down
            self.state[0] += 1
        elif action == 2 and self.state[1] < self.grid_size - 1:  # Right
            self.state[1] += 1
        elif action == 3 and self.state[1] > 0:  # Left
            self.state[1] -= 1

        if np.array_equal(self.state, self.goal):  # Check goal condition
            self.reward += 10
            self.done = True
        elif any(np.array_equal(self.state, hell) for hell in self.hell_states):  # Check hell states
            self.reward += -10
            self.done = False
            self.reset()  # Reset the environment when reaching hell
        else:  # Every other state
            self.reward += -0.05
            self.done = False

        self.info["Distance to goal"] = np.sqrt(
            (self.state[0] - self.goal[0])**2 + 
            (self.state[1] - self.goal[1])**2
        )
        
        return self.state, self.reward, self.done, self.info

    def render(self):
        """
        Renders the environment using Pygame.
        
        Returns:
            bool: True if rendering was successful, False if the quit event was detected.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == QUIT_KEY:  # Check if quit key is pressed
                    return False

        self.screen.blit(self.background_image, (0, 0))

        for x in range(0, SCREEN_WIDTH, CELL_SIZE):
            pygame.draw.line(self.screen, BLACK, (x, 0), (x, SCREEN_HEIGHT))
        for y in range(0, SCREEN_HEIGHT, CELL_SIZE):
            pygame.draw.line(self.screen, BLACK, (0, y), (SCREEN_WIDTH, y))

        for hell in self.hell_states:
            self.screen.blit(self.hell_image, (hell[1] * CELL_SIZE, hell[0] * CELL_SIZE))

        self.screen.blit(self.goal_image, (self.goal[1] * CELL_SIZE, self.goal[0] * CELL_SIZE))
        self.screen.blit(self.player_image, (self.state[1] * CELL_SIZE, self.state[0] * CELL_SIZE))

        pygame.display.update()
        return True

    def show_goal_reached(self):
        """
        Displays an animation and message when the goal is reached.
        """
        for scale in range(1, 6):  # Gradually scale up the image
            scaled_image = pygame.transform.scale(self.goal_reached_image, 
                                                  (self.goal_reached_rect.width * scale // 5, 
                                                   self.goal_reached_rect.height * scale // 5))
            scaled_rect = scaled_image.get_rect(center=self.goal_reached_rect.center)
            self.screen.blit(self.background_image, (0, 0))
            self.render()
            self.screen.blit(scaled_image, scaled_rect.topleft)
            pygame.display.update()
            pygame.time.wait(200)  # Wait 200 milliseconds between each scale step
        
        # Render the "Goal Reached" text
        text = self.font.render("Mr Olympia 2024!", True, RED)
        text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 100))
        self.screen.blit(text, text_rect)
        pygame.display.update()
        pygame.time.wait(2000)  # Display the message for 2 seconds

    def close(self):
        """
        Closes the Pygame window and quits Pygame.
        """
        pygame.quit()

def main():
    """
    Main function to run the custom environment.
    """
    env = CustomEnv(grid_size=7)

    observation, info = env.reset()
    print(f"Initial state: {observation}, Info: {info}")

    clock = pygame.time.Clock()
    done = False

    while not done:
        done = not env.render()

        # Take a random action
        action = env.action_space.sample()
        new_state, reward, done, info = env.step(action)
        print(f"New state: {new_state}, Reward: {reward}, Done: {done}, Info: {info}")

        if done and reward > 0:  # Check if goal is reached
            env.show_goal_reached()  # Show goal reached image and message

        clock.tick(10)  # Limit the loop to 10 frames per second

    env.close()

if __name__ == "__main__":
    main()
