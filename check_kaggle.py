import os
from kaggle.api.kaggle_api_extended import KaggleApi

def main():
    api = KaggleApi()
    api.authenticate()
    
    slug = "sergeistwpk/strawpick-segpoinnet-my-odin"
    print(f"Checking status for {slug}...")
    status = api.kernels_status(slug)
    print(f"Status: {status}")
    
    print(f"Trying to get output for {slug}...")
    # This usually downloads files to the current directory
    try:
        api.kernels_output(slug, path="logs_download")
        print("Output downloaded to 'logs_download'")
        files = os.listdir("logs_download")
        print(f"Files found: {files}")
    except Exception as e:
        print(f"Error getting output: {e}")

if __name__ == "__main__":
    main()
