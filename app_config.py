"""Project configuration项目配置."""

from dataclasses import dataclass
from typing import Dict, Literal

DatabaseType = Literal["custom"]

PERSIST_DIRECTORY = "db_storage"
OPENAI_BASE_URL = ""     # 需要自行填写的 OpenAI 兼容接口地址。

# 默认的 Hugging Face embedding 模型，可按需替换，修改后可能需要重建向量库
LOCAL_EMBEDDING_PATH = "aspire/acge_text_embedding"  


CUSTOM_COLLECTION_NAME = "custom_collection"


@dataclass
class CollectionConfig:
    name: str
    description: str
    collection_name: str


COLLECTIONS: Dict[DatabaseType, CollectionConfig] = {
    "custom": CollectionConfig(
        name="Custom Knowledge Base",
        description="Your fixed local knowledge from the data directory",
        collection_name=CUSTOM_COLLECTION_NAME,
    ),
}
