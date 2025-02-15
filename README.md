# RL Pendulum Physics Environment
This is a released package of a mathematically simulated pendulum that can be flexibly adapted for applying it as an RL environment. An advantage of this pendulum simulation is the flexibility of the parameter definition, *including the number of rods*, the rod masses, rod inertias, and more.

It is aimed to enable more flexible RL research on the difficult control problem of the cart-pendulum environment.

Best view [📒this notebook📒](https://drive.google.com/file/d/14xU5jiUNQYhsnEKenXVMwa_ODvL46T5Q/view?usp=drive_link) (.html file; 281 MB) and review the overall clean code to get all the information you need. 
Besides matplotlib animations and LaTeX math summaries, this package also supports rendering Plotly Animations, which are a lot faster to render for RL research, and easier to embed into [Panel](https://panel.holoviz.org/reference/index.html) UIs. Checkout [✨this interactive animation✨](https://drive.google.com/file/d/1ywLrFLX14-ld-VDXogV7GVBhv1LdpwNM/view?usp=sharing) (.html file; 56 MB).

I also provide an unfinished Panel UI prototype of a planned RL experiment platform that could help others build flexible UIs for experiment configurations and applications. It is out of my scope to finish this properly, however, this could provide a lot of value for other researchers. Here is a [🎞video demonstration🎞](https://drive.google.com/file/d/1xNR0xIT6O0zWD6OTT-0PAvU9RL7nfIQx/view?usp=sharing) (.mkv file; watchable w.o download) of the current Panel UI implementation.

Credit: The very good math derivations of [this notebook](https://colab.research.google.com/drive/1tonlB7P0w4EZv2eC8PMP9zO-FzwBizb_) for the triple cart-pendulum were reused for this project and abstracted to an n-rodded cart-pendulum class.

## Example Gallery

### Matplotlib + Plotly Animations

https://github.com/user-attachments/assets/8322b6c5-8e26-4855-8ac1-86297436e2fe

https://github.com/user-attachments/assets/313f5685-c03e-49e9-bb92-6604d2eebe7d

### The Current UI
![image](https://github.com/user-attachments/assets/29710f87-80ed-43e3-8e9c-8da6c9ed4646)
![image](https://github.com/user-attachments/assets/fd33b6e1-ac0d-49c6-bebf-ebd3713e3d99)
![image](https://github.com/user-attachments/assets/c6382672-4b8a-45b5-9293-b4cf2dcb3500)
![image](https://github.com/user-attachments/assets/7c89c8be-cfe5-453d-8e5f-10a71c4f7551)

## Setup

You generally have to follow the [setup_commands.txt](https://github.com/HenningBeyer/RL_Pendulum_Physics_Environment/blob/main/setup_commands.txt) to set up and launch the Panel UI. 

If you only want to run the RL pendulum in a notebook, it is recommended to only copy the self-contained directory [rl_pendulum/envs/cart_pendulum](https://github.com/HenningBeyer/RL_Pendulum_Physics_Environment/tree/main/rl_pendulum/envs/cart_pendulum), while leaving the other files. The other files won't be needed, and just give functionality to run the Panel UI.

🚧 Note that the [rl_pendulum/envs/rl_cart_pendulum](https://github.com/HenningBeyer/RL_Pendulum_Physics_Environment/tree/main/rl_pendulum/envs/rl_cart_pendulum) is not fully functional, but provides an extensive feature engineering pipeline. It is planned to reimplement the environment in JAX later and to use [Stoix](https://github.com/EdanToledo/Stoix) agents to solve it without much feature engineering.

