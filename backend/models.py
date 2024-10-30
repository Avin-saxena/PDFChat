from sqlalchemy import Column, Integer, String, Text, DateTime
from database import Base
from datetime import datetime

class Document(Base):
    __tablename__ = 'documents'

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), unique=True, nullable=False)
    original_filename = Column(String(255), nullable=False)
    content = Column(Text, nullable=True)
    upload_date = Column(DateTime, default=datetime.utcnow, nullable=False)
