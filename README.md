# Project Overview

This project investigated two strategies to enhance energy efficiency in automated machine learning: **warm-starting** and **few-shot learning**. The experiments used the **AutoGluon framework** to perform image classification tasks across three different datasets and utilized the **CodeCarbon tool** to measure the energy consumption during the model training. The results demonstrated that these methods significantly reduce energy usage, highlighting their practical benefits in deploying energy-efficient AutoML solutions.

# Datasets

- Coil-20 dataset contains grayscale images from 20 different objects. These objects are taken from multiple angles, and each object has 72 images from different angles, for a total of 1440 images. (Downloaded from: [COIL-20](https://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php))
- Plants-30 dataset contains images of 30 plants, 200 images for each plant, and a total of 6,000 images. (Downloaded from Kaggle: [Plants-30](https://www.kaggle.com/datasets/marquis03/plants-classification))
- Mammals-45 dataset contains images of 45 mammals. The number of samples in each category is not fixed (the minimum is 234 and the maximum is 356), with a total of 13,751 images. (Downloaded from Kaggle:[Mammals-45](https://www.kaggle.com/datasets/asaniczka/mammals-image-classification-dataset-45-animals))

# Setup

Each model was trained on the Tesla T4 GPU via the Google Colab platform. To ensure the reliability and accuracy of the experimental results, each experiment was conducted five times independently and the results were averaged.

# Results

## 1. Warm-starting Experiment

### - Using Default Model

The following table displays the performance comparison of the default model with and without the use of warm-starting technique on different datasets, covering metrics such as average accuracy, energy consumption, and runtime.

| Datasets     | Warm-starting | Avg. Accuracy | Avg. Energy Consumed (Wh) | Energy Drop % | Avg. Runtime | Runtime Drop % |
|--------------|---------------|---------------|---------------------------|---------------|--------------|----------------|
| Coil-20      | ✗             | 1.0           | 11.419                    | -             | 7m54s        | -              |
| Coil-20      | ✓             | 1.0           | 6.402                     | 43.9%         | 4m16s        | 45.3%          |
| Plants-30    | ✗             | 0.926         | 55.619                    | -             | 33m12s       | -              |
| Plants-30    | ✓             | 0.932         | 49.693                    | 10.6%         | 29m20s       | 11.6%          |
| Mammals-45   | ✗             | 0.978         | 154.431                   | -             | 1h27m19s     | -              |
| Mammals-45   | ✓             | 0.977         | 87.391                    | 43.5%         | 48m39s       | 44.3%          |

According to the results, warm-starting technique effectively reduced energy consumption and shortened runtime across datasets of various sizes, a finding consistently demonstrated in all tests. While the positive impact on accuracy varied slightly across datasets, overall, accuracy remained close to that of models not using warm-starting, and in some cases, it even improved.

### - Using Lightweight Model

In order to further reduce energy consumption, this experiment replaced the default model in AutoMM with the lightweight MobileNetv3-small network. The following table demonstrates the performance comparison of the MobileNetv3-small model with and without the use of warm-starting technique across different datasets. Other experimental conditions and settings remained consistent with those of the previous experiment.

| Datasets   | Warm-starting | Avg. Accuracy | Avg. Energy Consumed (Wh) | Energy Drop % | Avg. Runtime | Runtime Drop % |
|------------|---------------|---------------|---------------------------|---------------|--------------|----------------|
| Coil-20    | ✗             | 1.000         | 2.599                     | -             | 1m36s        | -              |
| Coil-20    | ✓             | 0.976         | 1.985                     | 23.63%        | 1m15s        | 21.88%         |
| Plants-30  | ✗             | 0.833         | 31.518                    | -             | 24m          | -              |
| Plants-30  | ✓             | 0.843         | 31.333                    | 0.59%         | 23m52s       | 0.56%          |
| Mammals-45 | ✗             | 0.907         | 59.231                    | -             | 42m58s       | -              |
| Mammals-45 | ✓             | 0.907         | 48.638                    | 17.89%        | 34m35s       | 19.40%         |

The data in the table revealed an important finding: replacing the default model, which has a larger number of parameters, with a lightweight network possessing fewer parameters, the consumption of computing resources could be further reduced.

In summary, the warm-starting method combined with lightweight networks, especially in resource-constrained application scenarios, can not only maintain or even improve model performance but also, more importantly, provide an effective way to reduce energy consumption and runtime.

## 2. Few-shot Learning Experiment

The following table provides a detailed comparison of the performance of few-shot classifiers and default classifiers in terms of accuracy, F1_macro, energy consumption, and runtime.

| Dataset     | Classifier | Avg. Accuracy | F1_macro | Avg. Energy Consumed (Wh) | Avg. Runtime  |
|-------------|------------|---------------|----------|---------------------------|---------------|
| Coil-20     | Few-shot   | 0.99          | 0.99     | 2.118                     | 1m41s         |
| Coil-20     | Default    | 0.85          | 0.88     | 8.549                     | 5m42s         |
| Plants-30   | Few-shot   | 0.80          | 0.83     | 3.268                     | 3m4s          |
| Plants-30   | Default    | 0.82          | 0.81     | 17.079                    | 11m26s        |
| Mammals-45  | Few-shot   | 0.81          | 0.83     | 4.819                     | 3m18s         |
| Mammals-45  | Default    | 0.87          | 0.88     | 15.204                    | 10m50s        |

On the Coil-20 dataset, the accuracy and F1_macro score of the few-shot learning method both reached 0.99, significantly better than the 0.85 and 0.88 of the default classifier. This shows that for simple datasets with fewer categories and samples, few-shot learning technique can effectively utilize limited data to identify key features, thereby achieving efficient learning.

Across all datasets, the energy consumption of the few-shot learning method was far less than that of the default classifier, highlighting its advantage in energy efficiency. Particularly on the Plants-30 dataset, the energy consumption of the few-shot classifier was only 19.1% of the default classifier (3.268Wh compared to 17.079Wh).





