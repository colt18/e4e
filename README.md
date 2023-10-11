Copy the "e4e_ffhq_encode.pt" and "face_landmarker_v2_with_blendshapes.task" under checkpoints dir.
Put your image file under images/input dir. There should be only one image file.

Run the following command. You can specify the degree as minus or plus.

```
python scripts/inference.py --degree 10
```
