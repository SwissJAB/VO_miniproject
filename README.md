# VO_miniproject

The goal of this mini-project is to implement a simple, monocular, visual odometry (VO) pipeline with the most essential features: initialization of 3D landmarks, keypoint tracking between two frames, pose estimation using established 2D ↔ 3D correspondences, and triangulation of new landmarks.

---

## Video Playlist

[https://www.youtube.com/playlist?list=PLYPB14LfGrkPhpV88stzWYthXyCQFvqRH](https://www.youtube.com/playlist?list=PLYPB14LfGrkPhpV88stzWYthXyCQFvqRH)

---

## How to Run the Setup

1. **Activate the conda environment** with the `environment.yaml` file:

    ```bash
    conda env create -f environment.yaml
    conda activate <env_name>
    ```

2. **Include a "Datasets" folder** in the root directory of the project such that "Code" and "Datasets" are in the same directory:

    ```
    root/
    ├── Code/
    └── Datasets/
    ```

3. **Set up dataset folder names**:
    - Ensure the dataset folders within `Datasets` are named:
      - `parking`
      - `kitti`
      - `malaga`
    - You can adjust these names in the `config.yaml` files as needed.

4. **Run the main script** from the root directory of the project:

    ```bash
    python Code/clean_vo.py
    ```

5. This will:
    - Launch the **parking dataset**.
    - Open two windows:
      - One for the continuous operation.
      - One showing the matches between the initialization frames for Structure-from-Motion (SFM).

6. **PRESS Q once** to close the SFM matches image and show the visualization.

7. **Adjust the configuration** in the `clean_vo.py` class at the bottom of the script. For example:

    ```python
    if __name__ == "__main__":
        pipeline = VisualOdometryPipeline(config_path='Code/configs/config_malaga.yaml')
        pipeline.run()
    ```
