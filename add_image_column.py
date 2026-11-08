#!/usr/bin/env python
"""Script để thêm cột image vào table attention_logs nếu chưa tồn tại."""

import os
import sys

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FLASK_DIR = os.path.join(SCRIPT_DIR, "flask_app")
sys.path.insert(0, FLASK_DIR)

os.chdir(SCRIPT_DIR)

from models import db
from app import create_app

def main():
    app = create_app()
    with app.app_context():
        try:
            # Kiểm tra xem cột image có tồn tại không
            inspector = db.inspect(db.engine)
            columns = [c['name'] for c in inspector.get_columns('attention_logs')]
            
            if 'image' not in columns:
                print('Adding image column to attention_logs table...')
                # SQLAlchemy 1.4+ không dùng engine.execute, phải dùng connection
                with db.engine.connect() as conn:
                    conn.execute(db.text('ALTER TABLE attention_logs ADD COLUMN image VARCHAR(255) NULL'))
                    conn.commit()
                print('✓ Column added successfully')
            else:
                print('✓ image column already exists')
        except Exception as e:
            print(f'✗ Error: {type(e).__name__}: {e}')
            sys.exit(1)

if __name__ == '__main__':
    main()
