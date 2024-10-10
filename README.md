# ADE-OoD: A Benchmark for Out-of-Distribution Detection Beyond Road Scenes
Official repository for the ADE-OoD benchmark, containing the evaluation code and annotation tool (coming soon). 

## :inbox_tray: Download
The data is available for download at the [project page](ade-ood.github.io).

## :chart_with_upwards_trend: Evaluation
This repo contains some utilities for evaluating your method on the ADE-OoD benchmark, without the need for other repositories.

These tools are also showcased in `example.py`.

#### Loading OoD scores from disk
If you already computed the scores for your method, you can use `evaluation.ade_ood_eval_with_scores_from_disk`:
```python
ap, fpr95 = ade_ood_eval_with_scores_from_disk(
    'SCORES_PATH',
    ade_ood_path='ADE_OOD_PATH', 
    scores_suffix='_my_method.npy'
    )
```
This requires that the scores are saved in 2D numpy arrays, and named like the original ADE-OoD images (plus an optional suffix).

#### Using a callback
If you want to pass your method as a callback, you can use `evaluation.ade_ood_eval_with_callback`:
```python
ap, fpr95 = ade_ood_eval_with_callback(
    your_method,
    data_preprocess_callback=your_preprocessing, 
    ade_ood_path='ADE_OOD_PATH'
    )
```
This requires your method to receive the input as its only positional, and return the scores as 2D torch/numpy arrays.

#### Explicit evaluation
For maximum flexibility you can use the `evaluation.ADEOoDDataset` and `evaluation.StreamingEval` classes directly, like this:
```python
ade_ood = ADEOoDDataset(os.path.expandvars('ADE_OOD_PATH'))
evaluator = StreamingEval(ade_ood.ood_idx)
for img, gt, img_name in ade_ood:
    img = your_preprocessing(img)
    score_map = your_method(img)
    evaluator.add(score_map, gt)
ap, fpr95 = evaluator.get_ap(), evaluator.get_fpr95()
```

## :white_check_mark: TODO
- Publish annotation tool

## :mortar_board: Citations
If you use this benchmark in your research, please cite the following papers:

```
@inproceedings{GalessoECCV2024,
      Title = {Diffusion for Out-of-Distribution Detection on Road Scenes and Beyond},
      Author = {Silvio Galesso and Philipp Schr\"oppel and Hssan Driss and Thomas Brox},
      Booktitle = {ECCV},
      Year = {2024}
      }
```
```
@InProceedings{Zhou_2017_CVPR,
        author = {Zhou, Bolei and Zhao, Hang and Puig, Xavier and Fidler, Sanja and Barriuso, Adela and Torralba, Antonio},
        title = {Scene Parsing Through ADE20K Dataset},
        booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        month = {July},
        year = {2017}
        }  
```
```
      @article{OpenImages,
        author = {Alina Kuznetsova and Hassan Rom and Neil Alldrin and Jasper Uijlings and Ivan Krasin and Jordi Pont-Tuset and Shahab Kamali and Stefan Popov and Matteo Malloci and Alexander Kolesnikov and Tom Duerig and Vittorio Ferrari},
        title = {The Open Images Dataset V4: Unified image classification, object detection, and visual relationship detection at scale},
        year = {2020},
        journal = {IJCV}
      }
```

