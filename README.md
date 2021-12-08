# Motion-Deblurring-with-Real-Events

This repository contains the codes and models for the paper "Motion Deblurring with Real Events" [[Paper\]](https://openaccess.thecvf.com/content/ICCV2021/papers/Xu_Motion_Deblurring_With_Real_Events_ICCV_2021_paper.pdf)  [[Supplementary\]](https://drive.google.com/file/d/1ftUon_OCQQw9ZdeGkiDM75-65EMAFnnK/view?usp=sharing) [[Project Page\]](http://dvs-whu.cn/projects/red/) 

If you use the code, models or data for your research, please cite us accordingly:

```
@inproceedings{xu2021motion,
  title={Motion Deblurring with Real Events},
  author={Xu, Fang and Yu, Lei and Wang, Bishan and Yang, Wen and Xia, Gui-Song and Jia, Xu and Qiao, Zhendong and Liu, Jianzhuang},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={2583--2592},
  year={2021}
}
```



## Pretrained Models

You can download the pretrained RED-RBE model from [here](https://drive.google.com/file/d/1xBbhMwqPR5p2Fs6VrGR0HagBzVOtNSD7/view?usp=sharing)



## Prerequisites & Installation

This code has been tested with CUDA 11.0 and Python 3.6.

```
pip install -r requirements.txt
```



## Get Started

You can download a sample sequence from this [link](https://drive.google.com/file/d/1kHgel64IRQF6dJmYVFlSnwagK0FPCBxW/view?usp=sharing), which is converted from [[HQF\]](https://drive.google.com/drive/folders/18Xdr6pxJX0ZXTrXW9tK0hC3ZpmKDIt6_).

Use the following command to test the neural network:

```
python test_deblur.py
```



# Credits

This code is based on the code available in [the slow-motion repo](https://github.com/MeiguangJin/slow-motion).  I am grateful to the authors for making the original source code available.



## Contact

We are glad to hear if you have any suggestions and questions.

Please send email to xufang@whu.edu.cn