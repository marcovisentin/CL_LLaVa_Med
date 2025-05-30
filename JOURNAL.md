The model checkpoint seems to be the pretrained and indstruct aligned model. It seems it is not the fine-tuned version on the three datasets. 

1. CATASTROPHIC FORGETTING:
- Download datasets in the right format
- Ensure model works well
- Fine-tune on the three datasets. Evaluate on the other two datasets against initial performance of the model.

21/03/2025

I am finetuning pretrained Med-LLava model on the three finetuning datasets. The data is unblanaced with path-vqa having around 19k data points, slake around 4k and vqa-rad 2k. 

Once trained I will evaluate its performance on the three test datasets.

Then I will further finetune the model on each individual dataset and test on all three datasets.
