import os
from dotenv import load_dotenv
load_dotenv()

FSUPABASE_URL= os.getenv("SUPABASE_URL")
FSUPABASE_KEY =os.getenv("SUPABASE_KEY")
FTABLE_NAME=os.getenv("TABLE_NAME")
FIMAGE_URL_PREFIX =os.getenv("FIMAGE_URL_PREFIX")



