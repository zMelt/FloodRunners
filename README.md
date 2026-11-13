# FloodRunners
Flood Analysis Probability for San Antonio

## GB10 Competition: Additional Information (e.g., running on local machine)

Materials for GB10 hackathons and student competitions. Follow these steps to prepare your computer for the Python notebooks.

You will install:

- **Anaconda:** includes Python and tools for managing your project’s software.
- **The OpenAI Python package:** lets your Python code send questions and images to an AI server. These workshop notebooks use it to connect to a GB10 server running Qwen.

For an overview of the event, download [the GB10 Hackathon primer](gb10-hackathon-primer-v4.html) and open the HTML file in your browser.

## 1. Install Anaconda

1. Go to the [official Anaconda download page](https://www.anaconda.com/download) and choose **Anaconda Distribution**.
2. Choose the installer for your computer. On a Mac with an M-series chip, choose **Apple Silicon / ARM64**. You can check your chip under **Apple menu → About This Mac**.
3. Install Anaconda:
   - **Windows:** open the downloaded `.exe` file. Choose **Just Me**, keep the default installation location, and leave **Add Anaconda to my PATH** unchecked.
   - **macOS:** open the downloaded `.pkg` file and follow the installer’s instructions.
   - **Linux:** follow the [official Linux installation instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) to run the downloaded `.sh` installer. If asked to initialize conda, choose **yes**.
4. After installation, open **Anaconda Prompt** from the Start menu on Windows, or open a new **Terminal** window on macOS or Linux.
5. Type this command and press **Enter**:

   ```sh
   conda --version
   ```

   A result such as `conda 26.x.x` means conda is available. Your version number may differ.

Need more help? See the official installation guides for [Windows](https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html) and [macOS](https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html).

**Older Intel Macs:** Anaconda has ended new Intel Mac package builds. See [Anaconda’s Intel Mac support notice](https://www.anaconda.com/blog/intel-mac-package-support-deprecation) and ask the organizer for a compatible setup if needed.

## 2. Install the OpenAI Python package

Run the following commands in **Anaconda Prompt or Terminal**, one line at a time. Do not enter them inside Python or a notebook cell.

First, create a separate environment for this workshop. An environment keeps this project’s Python and packages together.

```sh
conda create --name gb10 python=3.11 pip
conda activate gb10
```

If conda asks `Proceed ([y]/n)?`, type `y` and press **Enter**. Your prompt should now start with `(gb10)`.

Install the OpenAI package:

```sh
python -m pip install "openai>=1,<4"
```

This version range matches this project’s requirements. Using `python -m pip` installs the package into the Python environment you just activated.

Check that it installed correctly:

```sh
python -c "import openai; print('OpenAI package installed:', openai.__version__)"
```

You should see a version number without an error. This check does not contact an AI server or require an API key. See the [official OpenAI Python package documentation](https://developers.openai.com/api/reference/python) for more details.

## 3. Install the notebook tools and open the project

Download the full project folder from the repository or use the copy supplied by your organizer. If you download a ZIP file, extract it first. Keep the notebooks, Python files, and datasets together.

In the same prompt or terminal, use `cd` to open the folder containing this README and `requirements.txt`. Replace the example path below with your actual folder path; keep the quotation marks if it contains spaces.

Windows example:

```bat
cd /d "C:\Users\YourName\Downloads\GB10competition"
```

macOS or Linux example:

```sh
cd "$HOME/Downloads/GB10competition"
```

Then install the workshop packages and start JupyterLab:

```sh
python -m pip install -r requirements.txt
python -m jupyterlab
```

`requirements.txt` installs JupyterLab, the OpenAI package, and the image and data tools used in the notebooks. It is fine if the OpenAI package is already installed.

JupyterLab should open in your browser. If it does not, copy the local URL shown in your terminal into your browser. Keep the terminal open while working.

Double-click a notebook (`.ipynb`) in JupyterLab. If asked to choose a kernel, select **Python 3 (ipykernel)**. A kernel is the Python process that runs your notebook. Run cells from top to bottom with **Shift + Enter**.

## 4. Connect to the GB10 server

**Installing the OpenAI package does not install or start an AI model.** The model runs on the GB10 server provided for the workshop. Your computer must be able to reach that server; use the network or VPN specified by your organizer.

Ask the organizer for the server address, model name, and any required server API key.

- For the three Texas flood notebooks (`01_`, `02_`, and `03_`), update [vlm_config.py](vlm_config.py): `BASE_URL` is the supplied server URL ending in `/v1`, `MODEL` is the supplied model name, and `API_KEY` is the supplied server key. Use `NONE` only if the organizer confirms authentication is disabled.
- For `1_Introduction_to_Vision_Prompting.ipynb` and `2_Basic_Vision_Prompt_Structure.ipynb`, update the connection settings inside the notebook. Check each cell that creates an `OpenAI` client.
- Restart the notebook kernel after changing connection settings, then run the cells again.

You do not need an OpenAI account or OpenAI API key to use the workshop’s GB10 server. If you are doing a separate exercise with OpenAI’s hosted service, follow the [official OpenAI quickstart](https://developers.openai.com/api/docs/quickstart) for account and API key setup. Keep real API keys private and out of shared notebooks.

## Workshop files you may also need

- **Introductory notebooks:** some cells import a helper named `utils`. This checkout does not include `utils.py`; get the workshop helper file from the organizer and place it beside the notebooks if those cells report a missing `utils` module.
- **Texas flood notebooks:** get the workshop data from the organizer. The helper expects a `Texas_Flood_Student_Data` folder beside the project folder, containing `image_metadata.csv`, the images, and `observations.sqlite` for the weather exercise. Keep the supplied data layout intact.

Example folder layout for the Texas flood exercises:

```text
workshop/
├── GB10competition/
│   ├── README.md
│   ├── requirements.txt
│   └── 01_Image_Question_Answering.ipynb
└── Texas_Flood_Student_Data/
    ├── image_metadata.csv
    ├── observations.sqlite
    └── ... image files and folders supplied by the organizer
```

**Optional: direct image embeddings.** Only Method B in `02_Text_to_Image_Retrieval.ipynb` needs the extra packages below. Run this from the project folder with `gb10` activated, before starting JupyterLab:

```sh
python -m pip install -r requirements-embeddings.txt
```

This method downloads several GB of model files and runs on your computer. A compatible CUDA GPU is recommended; CPU execution is slower and needs substantial RAM. You can skip this installation for the other exercises.

## Returning to the workshop later

Open Anaconda Prompt or Terminal, change to the project folder, and run:

```sh
conda activate gb10
python -m jupyterlab
```

You only need to create the environment and install the packages once. To stop JupyterLab, save your notebooks, return to its terminal, and press **Ctrl + C**. Confirm shutdown if prompted.

## Common problems

| Problem | What to do |
| --- | --- |
| `conda` is not recognized or is not found | On Windows, use **Anaconda Prompt**. On macOS/Linux, close and reopen Terminal; if needed, follow the shell initialization instructions in your installation guide. |
| `No module named openai` | Run `conda activate gb10`, then repeat the package installation. Launch JupyterLab from that same environment. |
| A package is installed, but the notebook cannot import it | Check the notebook’s Python environment with `import sys; print(sys.executable)`. It should point to `gb10`. Restart JupyterLab from the activated environment if needed. |
| `requirements.txt` or an image cannot be found | Check that you opened the correct project folder and extracted the required data. For Texas flood data or `utils.py`, see the workshop files section above. |
| Connection error or timeout | Check the server URL, workshop network/VPN, and whether the organizer’s GB10 server is running. |
| Authentication error | Ask the organizer for the correct GB10 server key and update the relevant connection settings. |
