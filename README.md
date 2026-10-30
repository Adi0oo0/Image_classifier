# Image Classification Model
#### The project is about creating a prediction about a given image of it being AI generated or real!

# Project Structure
    data
        -test
            --ai
                ---*.jpg
            --real
                ---*.jpg
        -train
            --ai
                ---*.jpg
            --real
                ---*.jpg
    eval.py
    main.py
    requirements.py
    README.md
    *.pt

# Root Architecture and stack info
- Ive used `pytorch` along with `torchvision`
- I also used `pillow` just for `Image.open()` to read the data in an implicit way, i dont have to figure it out, like anything.
- The project is heavly based on the <a href="https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html" target="_blank">torch classifier documentation</a>.
- Some Standard values are used such as `IMAGENET_MEAN` and `IMAGENET_STD` from online sources, open by **`IMAGENET`** Dataset.

---
# [REMARK]


**You'll get a clear idea of what i've coded and where i have used Claude from the `commit history`.**

# Also
The model achieves **93.2%** accuracy on its dataset which it has been somewhat finetuned on.
- The model i've used is ResNet50 (architecture and weights) and dataset was from kaggle ai-vs-real dataset.

- ### It cannot be said that model is fine-tuned as i've just tweaked the last `linear layer` to output only 2 labels.

# Problems
- The dataset is not of good quality (224x224), that is the reason the model lags on real world data.
- The model not the claude version but mine, processed frozen layers aswell which was a huge inefficiency, and since claude generated the faster code i'm still not fully convinced how it works.
