import os
from dotenv import load_dotenv
load_dotenv()

FSUPABASE_URL= os.getenv("FSUPABASE_URL")
FSUPABASE_KEY =os.getenv("FSUPABASE_KEY")
FTABLE_NAME=os.getenv("FTABLE_NAME")
FIMAGE_URL_PREFIX =os.getenv("FIMAGE_URL_PREFIX")
