from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time

options = Options()
# options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--window-size=1200x600")
options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome(options=options)
driver.get("https://trex-runner.com/")

time.sleep(2)

# Start the game (simulate spacebar)
driver.execute_script("document.dispatchEvent(new KeyboardEvent('keydown', {'keyCode': 32}))")

# Let the game run for a few seconds
time.sleep(5)

# Extract some base data (dist, end of game, x and y)
distance = driver.execute_script("return Runner.instance_.distanceRan")
crashed = driver.execute_script("return Runner.instance_.crashed")
tRex_x = driver.execute_script("return Runner.instance_.tRex.xPos")
tRex_y = driver.execute_script("return Runner.instance_.tRex.yPos")

# Extract obstacles
obstacle_data = driver.execute_script("""
    const obs = Runner.instance_.horizon.obstacles;
    if (obs.length > 0) {
        const o = obs[0];
        return {
            type: o.typeConfig.type,
            xPos: o.xPos,
            yPos: o.yPos,
            width: o.typeConfig.width,
            height: o.typeConfig.height
        };
    }
    return null;
""")

# Res
print("Game Snapshot")
print(f"Distance Ran: {distance:.2f}")
print(f"Dino Position: x={tRex_x}, y={tRex_y}")
print(f"Crashed: {crashed}")

if obstacle_data:
    print("\nFirst Obstacle")
    for k, v in obstacle_data.items():
        print(f"{k}: {v}")
else:
    print("\nNo obstacles detected yet.")

driver.quit()
