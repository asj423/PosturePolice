# PosturePolice

Posture Police is a group Hackathon poject that is built using Python, SQL and Arduino C++. The goal was too create a machine that could monitor and detect how good your posture is, then alert you if anything needed to be corrected

## Features

 - **Real-time posture tracking** - Using mediapipe and openCV for computer vision, we are able to calculate the angles of your joints to decide how good your posture is.
 - **End of session statistics** - Using a database, we are able to keep track of how long you slouch for and how long your posture is good for.
 - **Infographics** - The user can watch their posture and see the angles between their joints to determine what is needed to be done to ensure good posture.

## My Contribution

My main job was to ensure that the main posturedetection.py was working. This meant I had to:
 - Create code to locate the persons joints
 - Calculate the angles between them
 - Classify them into good posture or bad posture
I also helped with the code for the arduino, ensuring it worked as expected so that the bot could alert the user as to when they're posture is bad.

## Tech Stack

 - Python
 - SQLite
 - Arduino C++
 - Mediapipe
 - OpenCV
 - Tkinter
