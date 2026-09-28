# Experiment log

Test = official ISIC 2018 Task 3 test set (1511 images). Val = our lesion-grouped 20% split of HAM10000.
Model selection uses val balanced accuracy; test is only looked at once per experiment.

| Date | Run | Model | Val bal_acc | Test bal_acc | Test macro AUC | Test mel sens / spec | Test malignant AUC | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-09-28 | `20260928_0249_baseline_effb0` | EfficientNet-B0, 224px | 0.776 (epoch 13 of 28) | 0.734 | 0.957 | 0.737 / 0.892 | 0.929 | MVP baseline, 10 min on RTX 3060 |

## Baseline notes (2026-09-28)

- Val peaked around epoch 10 of fine-tuning; train loss kept falling to 0.08 while val stayed flat at 0.73-0.78: overfitting in the second half.
- Main errors on test: 25/171 melanomas called nevus (missed), 97/908 nevi called melanoma (false alarms), benign keratosis confused with melanoma (35/217).
- Malignant vs benign (mel + bcc + akiec): sensitivity 0.831 at the argmax decision, AUC 0.929.

![Baseline confusion matrix on test](figures/baseline_effb0_confusion_test.png)
