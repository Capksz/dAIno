import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import threading
import subprocess
import os
import glob
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service


class DinoEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.driver = self._init_browser()
        self._start_game()
        self.num_frames = 0
        self.last_distance = 0
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.array([0, 6, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([100, 13, 600, 100, 600, 100], dtype=np.float32),
            dtype=np.float32
        )

        # Start the screenshot thread
        self.screenshot_thread = threading.Thread(target=self._capture_screenshots, daemon=True)
        self.screenshot_thread.start()
    def _make_video(self):
      cmd = [
          "ffmpeg",
          "-y",
          "-framerate", "7",
          "-i", f"static/screenshot_%04d.png",
          "-vframes", str(self.num_frames), 
          "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
          "-c:v", "libx264",
          "-pix_fmt", "yuv420p",
          "static/output.mp4"
      ]
      subprocess.run(cmd)
    
    def _capture_screenshots(self):
        self.num_frames = 0
        while 1:
            try:
                timestamp = int(time.time())
                screenshot_path = f"static/screenshot_{self.num_frames:04d}.png"
                self.driver.save_screenshot(screenshot_path)
                self.num_frames += 1
            except Exception as e:
                print(f"Error capturing screenshot: {e}")
                break

    def _init_browser(self):
        
        chrome_options = Options()
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--headless")


        driver = webdriver.Chrome(options=chrome_options)


        driver.get("https://trex-runner.com/")
        
        # time.sleep(2)
        return driver

    def _start_game(self):
        self.driver.execute_script("""
                document.dispatchEvent(new KeyboardEvent('keydown', {keyCode: 32}));
                document.dispatchEvent(new KeyboardEvent('keyup', {keyCode: 32}));
            """)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.driver.execute_script("Runner.instance_.restart()")
        time.sleep(0.05)
        self.last_distance = self.driver.execute_script("return Runner.instance_.distanceRan")
        return self._get_game_state()["obs"], {}

    def step(self, action):
        self._send_action(action)
        time.sleep(0.1)

        state = self._get_game_state()
        obs = state["obs"]
        done = state["crashed"]
        trex_width = state["trex_width"]
        trex_x = state["trex_x"]

        trex_y, trex_speed, obs1_dist, obs1_y, obs2_dist, obs2_y = obs

        distance_to_obstacle_1 = obs1_dist - (trex_width + trex_x)
        distance_to_obstacle_2 = obs2_dist - (trex_width + trex_x)
        distance_to_obstacle = distance_to_obstacle_1 if distance_to_obstacle_1 > 0 else distance_to_obstacle_2

        reward = 1
        if done:
            reward = -100
        # else:
        #     if int(action) == 0:
        #         if 0 < distance_to_obstacle <= 50:
        #             reward -= 8
        #
        #     elif int(action) == 1 or int(action) == 2:
        #         if 0 < distance_to_obstacle <= 50:
        #             reward += 10
        #         else:
        #             reward -= 8

        # # X-axis overlap + Y-axis non-overlap = successful dodge
        # if trex_x + trex_width > obs1_x and trex_x < obs1_x + obs1_width:
        #     if trex_y + trex_height < obs1_y or trex_y > obs1_y + obs1_height:
        #         reward += 5  # dodged it
        #     else:
        #         reward -= 3  # overlapping, maybe risky

        # print(action, reward)
        return obs, reward, done, False, {}

    def _get_game_state(self):
        try:
            distance = self.driver.execute_script("return Runner.instance_.distanceRan")
            crashed = self.driver.execute_script("return Runner.instance_.crashed")
            trex_x = self.driver.execute_script("return Runner.instance_.tRex.xPos")
            trex_y = self.driver.execute_script("return Runner.instance_.tRex.yPos")
            width = self.driver.execute_script("return Runner.instance_.tRex.config.WIDTH")
            height = self.driver.execute_script("return Runner.instance_.tRex.config.HEIGHT")
            ducking = self.driver.execute_script("return Runner.instance_.tRex.ducking")
            duck_width = self.driver.execute_script("return Runner.instance_.tRex.config.WIDTH_DUCK")
            duck_height = self.driver.execute_script("return Runner.instance_.tRex.config.HEIGHT_DUCK")
            trex_jump = self.driver.execute_script("return Runner.instance_.tRex.jumping")
            trex_duck = self.driver.execute_script("return Runner.instance_.tRex.ducking")
            trex_jump_velocity = self.driver.execute_script("return Runner.instance_.tRex.jumpVelocity")
            current_speed = self.driver.execute_script("return Runner.instance_.currentSpeed")
            obstacles = self.driver.execute_script("""
                const obs = Runner.instance_.horizon.obstacles;
                if (obs.length > 0) {
                    return obs.map(o => ({
                        xPos: o.xPos,
                        yPos: o.yPos,
                        width: o.width,
                        height: o.typeConfig.height,
                        type: o.typeConfig.type
                    }));
                }
                return [];
            """)

            if trex_duck:
                width = duck_width
                height = duck_height

                # Sort obstacles by xPos (left boundary)
            obstacles.sort(key=lambda o: o['xPos'])

            # Skip all obstacles that are fully behind the Dino
            relevant_obstacles = [
                obs for obs in obstacles
                if obs['xPos'] + obs['width'] >= trex_x  # if obstacle not fully passed
            ]

            # print(obstacles)

            # Get up to 2 relevant obstacles
            obs1 = relevant_obstacles[0] if len(relevant_obstacles) > 0 else {"xPos": 600, "yPos": 0, "width": 0,
                                                                              "height": 0, "type": 3}
            obs2 = relevant_obstacles[1] if len(relevant_obstacles) > 1 else {"xPos": 600, "yPos": 0, "width": 0,
                                                                              "height": 0, "type": 3}
            obs_type_conversion = {"CACTUS_SMALL": 0, "CACTUS_LARGE": 1, "PTERODACTYL": 2, 3: 3}
            obs1["type"] = obs_type_conversion[obs1["type"]]
            obs2["type"] = obs_type_conversion[obs2["type"]]
            obs_array = np.array([
                trex_y, current_speed,
                obs1['xPos'] - trex_x, obs1['yPos'],
                obs2['xPos'] - trex_x, obs2['yPos'],
            ], dtype=np.float32)

            # print(obs_array)
            return {"obs": obs_array, "crashed": crashed, "distance": distance, "trex_x": trex_x, "trex_width":width}

        except Exception as e:
            print("JS read error:", e)
            return {"obs": np.zeros(12, dtype=np.float32), "crashed": True}

    def _send_action(self, action):
        if action == 0:
            pass

        elif action == 1:  # jump
            self.driver.execute_script("""
                document.dispatchEvent(new KeyboardEvent('keydown', {keyCode: 32}));
            """)
            self.driver.execute_script("""
                document.dispatchEvent(new KeyboardEvent('keyup', {keyCode: 32}));
            """)

        elif action == 2:
            self.driver.execute_script("""
                document.dispatchEvent(new KeyboardEvent('keydown', {keyCode: 40}));
            """)
            # time.sleep(0.2)
            self.driver.execute_script("""
                document.dispatchEvent(new KeyboardEvent('keyup', {keyCode: 40}));
            """)
    def close(self):
     
        self.driver.quit()
        self.screenshot_thread.join()
        self._make_video()
# 