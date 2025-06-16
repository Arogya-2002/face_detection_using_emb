from src .constants import *

from dotenv import load_dotenv
import os

load_dotenv()


class ConfigEntity:
    def __init__(self):
        self.ssupabase_url= FSUPABASE_URL
        self.supabase_key =FSUPABASE_KEY
        self.table_name=FTABLE_NAME




class FaceIdentifierConfigEntity:
    def __init__(self, config=ConfigEntity()):
        self.supabase_url = config.ssupabase_url
        self.supabase_key = config.supabase_key
        self.table_name = config.table_name
