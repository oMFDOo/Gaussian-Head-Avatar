## 요구 데이터 삽입
공식 문서의 [Datasets](https://github.com/oMFDOo/Gaussian-Head-Avatar/tree/main?tab=readme-ov-file#datasets)에 따르면 이곳에 [NeRSemble](https://tobias-kirschstein.github.io/nersemble/) 데이터를 넣어야한다. <br>
이용을 위해서는 비영리 기관 메일로 요청을 보내야하는데, 테스트를 위해 제작자가 친절히 데모용 데이터를 넣어주었다. 하지만 이 또한 약관에 동의하여 요청을 보내었다는 전제를 바탕으로 해야한다. 

<br><br>

임시로 제공하는 [mini demo dataset](https://drive.google.com/file/d/1OddIml-gJgRQU4YEP-T6USzIQyKSaF7I/view?usp=drive_link)을 이용하여 압축을 풀어 이곳에 넣게 된다. <br>
그렇다면 파일 구조는 아래와 같아질 것이다.
```
031
 - background
   > images_220700191.jpg
   > images_221501007.jpg
   > images_222200036.jpg
   ...
 - cameras
   > 0000
     + camera_220700191.npz
     + camera_221501007.npz
     + camera_222200036.npz
     ...
   > 0010
   > 0020
   ...
   > 1800
 - images
   > 0000
     + image_220700191.jpg
     + image_lowres_220700191.jpg
     ...
 - landmarks
   > 0000
     + camera_220700191.npy
     + camera_221501007.npy
     + camera_222200036.npy
     ...
 - params
   > 0000
     + lmk_3d.npy
     + params.npz
     + vertices.npy
036
```

<br>

---

<br>

공식 데이터를 받은 이후에는 [Multiview-3DMM-Fitting](https://github.com/YuelangX/Multiview-3DMM-Fitting)에서 제공하는 데이터 전처리 코드를 따라야한다. <br>
공식 문서의 [Datasets](https://github.com/oMFDOo/Gaussian-Head-Avatar/tree/main?tab=readme-ov-file#datasets)를 확인해보도록 하자