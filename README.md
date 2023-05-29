# Inertial navigation transformer

## Introduction
Inertial navigation relies on measurements from an inertial measurement unit (IMU) - a device typially consisting of a gyroscope, an accelerometer and a magnetometer. The IMU is mounted on a moving object, and the measurements are used to estimate the position and orientation of the object.

<figure>
  <img src="https://github.com/tonton-golio/inertial_navigation_transformer/blob/main/assets/IMU_interpretation_flowchart.png?raw=true" alt="IMU interpretation flowchart">
  <figcaption>IMU interpretation flowchart. Angular rates are measured by the gyroscope, bias corrected and intergrated to obtain the change in orientation. Accelerations are rotated into the world fram and intergrated twice to obtain position.</figcaption>
</figure>

## Quaternion update
The quaternion update is implemented analyically in `utils.Theta(angular_velocity, timestepsize)`. This is applied in  `playground_notebooks/quaternion_update_anton.ipynb`, along with a neural network implementation of the same function. The outputs of both methods is shown below.

<figure>
  <img src="https://github.com/tonton-golio/inertial_navigation_transformer/blob/main/assets/quat_nn.png?raw=true" alt="IMU interpretation flowchart">
  <figcaption>Dense neural network with 10 layers of width 64. The network was given a quaternion and 40x3 gyroscope reading and asked to predict the 41st quaternion.</figcaption>
</figure>

**Open tasks on quaternion update**
- [] emulate what Anton has done
- [] this still needs testing on alternative dataset, or entirely unseen data. 
- [] optimize parameters (network and hyper); number of layers, widths, learning rate, batch size, number of timesteps between truths, etc.
- [] include magnetometer and/or accelerometer data.
- [] implement RNN.
- [] implement LSTM.
- [] implement transformer network.

## Body frame to world frame
An initial orientation is required to interpret the measurements from the IMU. The world frame is defined by the north, east and down directions, obtained by the direction of the magnetic field and the direction of gravity (and their cross product).
> This is implemented in `playground_notebooks/bodyFrame_to_worldFrame.ipynb`.

**Open tasks on body2world**
- [] instead of just using a single timestep take the average of the first couple, and see if accuracy is improved.
- [] implement a neural network which takes the first N timesteps of `synced/acce` and `synced/magnet` and predicts the orientation.
  - [] does it help the network, if we add `synced/gyro` as well?

## Acceleration to position
The acceleration is rotated into the world frame, and integrated twice to obtain position. However this method is very sensitive to noise, and the position estimate drifts quickly. 

**Open tasks on acc2pos**
- [] implement simple integration of acceleration to position, and insert result here
- [] implement neural network which takes `synced/acce` and `synced/magnet` and true orientation and predicts the position.

## Data
The data is obtained from ... Below is a table of the data contents.

## Timeline
| Date | Task | Status | Notes |
| --- | --- | --- | --- |
| 05-15 | First meeting | Done | We got the data loaded and distributed initial tasks |
| 05-24 | excercise class | Done | Simon and Anton worked on the quaternion update and rewrote the README |

| 06-14 | deadline for project |  |  |


## Personal tasks
### Michael

| Date added| Task | date completed  | Status | Notes |
| --- | --- | --- | --- | --- |
| 05-15 | clone github + open data + watch introductory youtube videos | 05-15 | Done | |
| 05-15 | make overview of data contents |  | Single dataset done | missing full |
| 05-15 | Feed a NN: accelerions data along with tango ori, and predict positons | |  | |


### Anton
| Date added| Task | date completed  | Status | Notes |
| --- | --- | --- | --- | --- |
| 05-15 | make github + open data + watch introductory youtube videos | 05-15 | Done | |
| 05-15 | Implement analytical quaternion update | 05-15 | Done | See `playground_notebooks/quaternion_update_anton.ipynb` |
| 05-15 | Implement body frame to world frame | 05-15 | Done | See `playground_notebooks/bodyFrame_to_worldFrame.ipynb` |
| 05-16 | Implement simple Transformer network | 05-16 | Done | see `playground_notebooks/transformer_initial_anton.ipynb` |
| 05-15 | Implement NN for replacing `utils.Theta(angular_velocity, timestepsize)` | 05-24 | Done | |

### Adrian
| Date added| Task | date completed  | Status | Notes |
| --- | --- | --- | --- | --- |
| 05-15 | clone github + open data + watch introductory youtube videos | 05-15 | Done | |
| 05-15 | Implement NN for replacing `utils.Theta(angular_velocity, timestepsize)` |  |  | |

### Simon
| Date added| Task | date completed  | Status | Notes |
| --- | --- | --- | --- | --- |
| 05-15 | clone github + open data + watch introductory youtube videos | 12-05  | Done | |
| 25-05 | Go through and optimize Anton's NN | 28-05 |Done | |
| 28-05 | Implement LSTM | | | |

### Chris
| Date added| Task | date completed  | Status | Notes |
| --- | --- | --- | --- | --- |
| 05-15 | clone github + open data + watch introductory youtube videos |  |  | |
| 05-24 | Feed a NN: accelerions data along with tango ori, and predict positons | |  | |

### Unassigned
| Date added| Task | date completed  | Status | Notes |
| --- | --- | --- | --- | --- |


## Todo
- [] list of material for research/introduction to dealing with IMU data.
- [] Implement neural network or for replacing `utils.Theta(angular_velocity, timestepsize)`. This should use `tango/ori` as ground truth, and take `synced/gyro` as inputs. For now, set the initial orientation equal to the groundtruth. Subsequently, implement larger network which also takes `synced/acce` and `synced/magnet` to predict world frame, instead of getting initial orientation from ground truth.
    * consider advantages of different architectures: feed-forward, RNN, LSTM, transformer.
    * We found that initial orientation introduces an issue
    * A dysfynctional version of this has been added to `playground_notebooks/quaternion_update_anton.ipynb`
- [] Make table of data contents, so we get an overview of what we are working with.




## links to get into the topic:
* The problem with IMU position estimation https://www.youtube.com/watch?v=_q_8d0E3tDk
* basic kalman filter introduction: https://www.youtube.com/watch?v=5HuN9iL-zxU
* get NED values for acc: https://www.youtube.com/watch?v=T9jXoG0QYIA
* strapdown: https://www.youtube.com/watch?v=knroQcjAgPo
* Romberg integration: https://www.youtube.com/watch?v=HtkylnbJklE
* paper on coning and sculling (integration techniques) https://sci-hub.hkvisa.net/10.2514/2.4718# This paper shows us how to convert a coning algorithm to its sculling algorithm equivalent, or vice versa. (probably no need to look at this)
* https://sci-hub.hkvisa.net/10.3390/s111009182 -- a great introduction!
* inertial navigation primer: https://www.vectornav.com/resources/inertial-navigation-primer/math-fundamentals/math-coning

## Ideas
* throw raw acc, gyro and magnet at LSTM -> predict pose
  * hyperparameter-sweep
* transform raw inputs and measure the deacrease in training time, or increase in performance, or decrease in required NN size
* try transformer network
* Raw inegration for baseline
* kalman filter for better baseline (https://sci-hub.hkvisa.net/10.3390/s111009182)
* identify magnetic field disurbtions 
* single step (or few step) for neural network. 
  - Input vector is then: current state s_i (position and orientation) and imu readings d_{i:i+n}. 
  - Output is next state: s_n
* visualization tool (potentially 3d)
* streamlit app for visualization
