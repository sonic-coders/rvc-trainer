
<div align="center">

# 📚 TensorBoard: User Guide 📚

</div>

---

<div align="center">

## 🌟 Introduction 🌟

</div>

TensorBoard is a powerful tool for visualizing and monitoring the training process of machine learning models. It allows you to track metrics, visualize graphs, images, and even histograms. In this guide, we will take a detailed look at how to use TensorBoard to monitor model training progress, with a focus on the **`g/total`** metric. This tool is particularly useful for Generative Adversarial Network (**GAN**) models, but can also be applied to other types of models.

---

<div align="center">

## 🚀 TensorBoard Basics 🚀

</div>

### ⚙️ Setting Up TensorBoard

1. **Navigate to the "Scalars" Tab**:
   - After launching TensorBoard, go to the **"Scalars"** tab. Here you will find graphs of all your metrics, including **`g/total`**.
   - If there are too many graphs, use the search bar at the top of the interface to quickly find the one you need.

2. **Smoothing the Graph**:
   - Set the **smoothing** to **`0.950`** or **`0.987`**. This will help reduce noise and make metric changes clearer and more understandable.

3. **Automatic Data Refresh**:
   - Click the gear icon (⚙️) in the upper right corner.
   - Enable automatic data reload every **30 seconds** to see updates in real time.

4. **Working with Graphs**:
   - Below each graph, there are three buttons:
     - **First** — to enlarge the graph.
     - **Second** — to disable the Y-axis.
     - **Third** — to fit the data to the graph.
   - Uncheck the **"ignore outliers in chart scaling"** option to avoid graph distortion due to outliers.

### 📉 How to Find the Optimal Point?

The optimal point is the moment when the graph drops to its minimum value and does not repeat afterwards. During training, there may be several such points worth testing to find the best model and prevent overfitting.

- **How to identify the optimal point**:
  - Visually find the lowest point on the graph. Use the **`alt + mouse wheel`** combination to zoom in on the area of interest.
  - Hover your cursor over this point — a floating window with information about it will appear.
  - Note the number of **steps** at this point.
  - Navigate to the folder where models are saved and find the **epochs** corresponding to that step.
    > The model save format is usually: **`ModelName_e10_s100.pth`**, where **`e10`** is the number of epochs and **`s100`** is the steps.
  - Save the model and test it.

### 📊 Detailed Metric Analysis

In addition to **`loss/g/total`**, it is important to track other metrics such as **`loss/g/mel`**, **`loss/g/kl`**, and **`loss/d/total`**. If any of these values rise and do not decrease, it is a sign of overfitting. At the beginning of training, it is normal for **`d/total`** to rise and **`g/total`** to fall, but eventually they should stabilize. All metrics should decrease or fluctuate around similar values. For example, if **`g/mel`** starts to rise instead of falling, it indicates problems in the model. The **`g/mel`** graph shows the model's accuracy in reproducing the mel-spectrogram from your dataset, reflecting audio quality.

---

<div align="center">

## 🧠 Loss Metrics: What Do They Mean? 🧠

</div>

Loss metrics provide valuable feedback during GAN training, helping to improve the quality and realism of generated data. Monitoring these loss metrics together gives a comprehensive view of training progress. Successful training typically demonstrates a balance between decreasing generator loss (**`loss/g/total`**) and discriminator loss (**`loss/d/total`**), as well as decreasing feature matching loss (**`loss/g/fm`**), mel-spectrogram loss (**`loss/g/mel`**), and KL divergence loss (**`loss/g/kl`**). This indicates that the generator is producing increasingly realistic generated data while maintaining a similar internal structure and distribution to the real data.

Optimal values will vary depending on the specific task and dataset. However, by tracking trends in these loss metrics, you can gain valuable insights into training progress and identify potential issues that may arise.

> [!NOTE]
> ### (D) Discriminator Loss (`loss/d/total`):
> Discriminator loss measures the discriminator's ability to distinguish between real and generated data. It is typically formulated as a cross-entropy loss, where the discriminator is penalized for misclassifying real and generated samples.
>
> A decreasing discriminator loss indicates that the discriminator is getting better at distinguishing between real and generated data. This is usually a good sign, as it suggests that the generator is producing increasingly realistic generated data. However, if the discriminator loss becomes too low, it may indicate that the generator is simply memorizing the real data rather than learning to capture its underlying patterns.

> [!NOTE]
> ### (G) Generator Loss (`loss/g/total`):
> Generator loss measures the generator's ability to produce generated data that is indistinguishable from real data. It is typically formulated as an adversarial loss, where the generator is penalized for producing data that is easily classified as generated by the discriminator.
>
> A decreasing generator loss indicates that the generator is successfully learning to produce realistic generated data.

> [!NOTE]
> ### Feature Matching Loss (`loss/g/fm`):
> Feature matching loss encourages the generator to produce generated data that has similar intermediate feature representations to real data. This is achieved by comparing the intermediate activations of a feature extractor applied to both real and generated data.
>
> A decreasing feature matching loss indicates that the generator is producing generated data with intermediate feature representations similar to real data. This suggests that the generator is learning to capture the internal structure of the real data, which contributes to the realism of the generated data.

> [!NOTE]
> ### Mel-Spectrogram Loss (`loss/g/mel`):
> Mel-spectrogram loss compares the mel-spectrograms of real and generated data. Mel-spectrograms are a way of representing the frequency content of an audio signal, and this loss encourages the generator to produce generated data that sounds similar to real data.
>
> A decreasing mel-spectrogram loss indicates that the generator is producing generated data with a similar spectral distribution to real data. This is particularly important for audio generation tasks, as it ensures that the generated audio sounds similar to real audio.

> [!NOTE]
> ### KL Divergence Loss (`loss/g/kl`):
> KL divergence loss encourages the generator to produce generated data that has a similar latent variable distribution to real data. Latent variables are representations of the internal factors generating the data, and this loss ensures that the generator is not simply memorizing the real data but is learning to capture the underlying patterns in the data.
>
> A decreasing KL divergence loss indicates that the generator is producing generated data with a similar latent variable distribution to real data. This suggests that the generator is not simply memorizing the real data but is learning to capture the underlying patterns in the data, leading to more generalizable generated data.

---

<div align="center">

## 📘 Additional Information 📘

</div>

### Scalars:

- **`loss/g/total`**: Total generator loss. This measures how well the generator can fool the discriminator and produce realistic-sounding audio. A lower value is generally better.
- **`loss/d/total`**: Total discriminator loss. This loss reflects how well the discriminator can distinguish between real and generated audio. A higher value typically indicates a well-performing discriminator.
- **`learning_rate`**: The current learning rate of the optimizers for both the generator and discriminator. This value typically decreases over time, as defined by the learning rate scheduler.
- **`grad_norm_d`**: Gradient norm for the discriminator. A measure of the magnitude of gradients during training. It helps track whether gradients are becoming too large (which can cause instability) or too small (which can lead to slow learning).
- **`grad_norm_g`**: Gradient norm for the generator. Similar to **`grad_norm_d`**, this measures the magnitude of gradients for the generator.
- **`loss/g/fm`**: Feature matching loss. This loss encourages the generator to produce audio that matches feature maps extracted by the discriminator, promoting similar acoustic characteristics between real and generated audio.
- **`loss/g/mel`**: Mel-spectrogram loss. This loss measures the difference between the mel-spectrogram of the generated audio and the target mel-spectrogram. A lower value indicates a better match in spectral characteristics.
- **`loss/g/kl`**: Kullback-Leibler (KL) divergence loss. This loss measures the difference between the latent variable distributions generated by the encoder and the posterior encoder. A lower value indicates a better match between these distributions, contributing to a more stable and controllable model.
- **`loss/g/{i}`**: Individual generator losses for each discriminator layer. These losses provide a more detailed breakdown of generator performance at different time scales.
- **`loss/d_r/{i}`**: Individual discriminator losses for real audio at each discriminator layer. These losses show how well the discriminator performs on real audio at different scales.
- **`loss/d_g/{i}`**: Individual discriminator losses for generated audio at each discriminator layer. These losses indicate the discriminator's ability to distinguish generated audio from real audio at different time scales.

### Images:

- **`slice/mel_org`**: Visualization of the mel-spectrogram of the target audio segment.
- **`slice/mel_gen`**: Visualization of the mel-spectrogram of the generated audio segment.
- **`all/mel`**: Visualization of the mel-spectrogram of the entire target audio.

---

<div align="center">

## 🔍 Interpretation 🔍

</div>

- **Overall Trend**: Look for a decrease in the total generator loss (**`loss/g/total`**) and stability or an increase in the discriminator loss (**`loss/d/total`**).
- **Gradient Norm**: Monitor gradient norms to avoid them becoming excessively large or small.
- **Loss Components**: Analyze individual loss components (e.g., **`loss/g/mel`**, **`loss/g/kl`**) to understand how different aspects of the model are performing.
- **Mel-Spectrograms**: Compare mel-spectrogram images (**`slice/mel_org`** and **`slice/mel_gen`**) to visually assess the quality of the generated audio.
