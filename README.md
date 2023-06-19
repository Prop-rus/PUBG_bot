# PUBG Bot
This program is a Python-based bot designed for the game PlayerUnknown's Battlegrounds (PUBG). The bot is capable of automating various tasks to earn some game bonuses.

## Before You Begin
Before using this project, please take note of the following:

- This project is intended for individuals who have at least a basic understanding of Python programming language.
- You should be capable of independently installing the necessary dependencies, such as OpenCV and PyTorch, on a Windows operating system.

To ensure a smooth experience with the PUBG bot, please make sure you meet the prerequisites and have the required knowledge and skills. If you are new to Python or need assistance with installing the dependencies, there are various online resources, tutorials, and forums available to help you get started.
## Background
In the game PUBG, players are rewarded with additional bonuses and events for spending a significant amount of time in the game each day. However, simply launching matches and being AFK (Away From Keyboard) will not earn you Battle Points (BP) and Survivor Pass experience at the end of the match. To receive these rewards, players must perform at least one action, such as picking up an item or getting into any vehicle. By completing these actions, the game recognizes the player as active and credits them with bonuses.

This bot is specifically developed to automatically increase gameplay time without human intervention and to obtain bonuses based on the completion of the minimum in-game actions mentioned above. By automating certain tasks and actions, the bot ensures that players can accumulate game time efficiently and earn the rewards available for active participation.

## Features

### Demo
[![Watch the video](https://img.youtube.com/vi/mZVVpc4FQ8s/hqdefault.jpg)](https://youtu.be/mZVVpc4FQ8s)

### Match Initialization
The bot automatically initiates a match, handling the process of starting a new game.
### Map Detection
The bot can determine the current map being played, allowing for map-specific strategies and decision-making.
### Target Location Selection
The bot identifies and selects a target location for landing and distance from the flight path.
### Jump Timing Calculation
The bot calculates the optimal time to jump from the airplane, ensuring a precise landing near the target location.
### Landing and Navigation
The bot successfully lands in the vicinity of the target location and navigates towards it, utilizing image scanning techniques to detect any available vehicles.
### Vehicle Interaction
Upon detecting a vehicle, the bot directs itself towards it and boards the vehicle, thereby allowing the game to recognize the player as active. This action contributes to earning Battle Points, survival experience, and other in-game resources.
### Lobby Navigation
After completing a match, the bot automatically identifies and interacts with the lobby navigation buttons, facilitating a seamless transition to the next game.
### Bonus Collection
The bot collects bonuses for playing multiple matches in a day, maximizing resource accumulation through prolonged gameplay.

## Advanced Abilities
The bot possesses additional advanced abilities, including:
### Obstacle Avoidance
The bot is capable of intelligently maneuvering around obstacles on its path towards the target location, ensuring efficient navigation and minimizing delays.
### Lobby Restarting
In case of game errors or issues, the bot can autonomously restart the lobby, allowing for continuous gameplay without manual intervention.
### Loot Gathering
On maps where vehicles are scarce or fixed spawn locations are absent, the bot can gather loot, compensating for the lack of transportation resources and increasing the chances of getting rewarded by bonuses.

## Important Warning
Please read this section carefully before using the PUBG bot.

While the bot does not employ any hacks or bypass game rules and solely relies on in-game screenshots provided by users, it is important to note that the game's administration may penalize players for certain behaviors, such as being AFK (Away From Keyboard). The bot's actions, although automated and within the boundaries of image recognition, can still be interpreted as not actively participating in the game.

The author of this program does not assume any responsibility for potential penalties or consequences resulting from the usage of any part of the program by the game's administration.

It is strongly advised to exercise caution and discretion when using the bot to avoid any violations of the game's terms of service.

## Usage
To use the PUBG bot, follow these steps:

1. Ensure that Python is installed on your system.
2. Install the necessary dependencies by 
    - running pip install -r requirements.txt.
    - installing openCV library for python https://docs.opencv.org/4.x/d5/de5/tutorial_py_setup_in_windows.html
    - installing pyTorch library with CUDA https://pytorch.org/get-started/locally/
3. Download and extract the archive containing the pre-trained PyTorch YOLOv5 model and screenshots by the link https://drive.google.com/file/d/1vBmCYuLozMcIXTTTamSWD3_2TdedWP8t/view?usp=sharing.
4. Launch the bot script "main.py".
5. Configure the desired settings, such as map preferences, target location criteria, and advanced abilities.
6. Let the bot automate the gameplay, monitor its actions, and enjoy an enhanced PUBG experience.

Important Note: The bot has been trained on templates with a resolution of 2560 x 1440 and the Russian language. If your game resolution or language differs, you must adjust the resolution in configs/resolution_conf.py and prepare your own screenshots and templates following the same format as provided in the archive. Ensure that the newly obtained templates are named identically and replace the ones in the "screenshots" directory.

## Compatibility
The PUBG bot is fully functional and compatible with game version 23.2. It has been tested and verified to work seamlessly with this specific version of the game. However, it is important to note that game behavior and mechanics may change in future updates.

While the bot may require adjustments to adapt to newer game versions, you can use the existing codebase as a foundation and make necessary modifications to accommodate any changes or additions introduced in subsequent updates.

The provided code serves as a starting point, allowing you to enhance and customize the bot according to the evolving features and requirements of the game. Stay updated with the game's latest version and make the necessary changes to ensure continued functionality.

Please note that the bot's compatibility and support for future game versions may vary, and it is your responsibility to update and maintain the bot's codebase to align with the latest game updates.

## Acknowledgments
Special thanks to the developers of PUBG and the Python community for providing the necessary resources and tools to create this bot.

### MIT License

Copyright (c) 2023 Prop-rus

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.