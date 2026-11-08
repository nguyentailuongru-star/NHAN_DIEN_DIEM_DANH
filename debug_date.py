"""Debug script để kiểm tra date filtering."""
import os
import sys
from datetime import datetime, date, timedelta

sys.path.insert(0, 'flask_app')
from models import db, AttentionLog
from app import create_app

app = create_app()
with app.app_context():
    today = date.today()
    start_day = today - timedelta(days=6)
    start_dt = datetime.combine(start_day, datetime.min.time())
    end_dt = datetime.combine(today, datetime.max.time())
    
    print(f'Today: {today}')
    print(f'Start: {start_dt}')
    print(f'End: {end_dt}')
    print()
    
    all_recs = db.session.query(AttentionLog).all()
    print(f'Total records: {len(all_recs)}')
    for r in all_recs:
        in_range = start_dt <= r.time <= end_dt
        print(f'  - {r.time} -> in_range={in_range}')
    
    count = db.session.query(AttentionLog).filter(
        AttentionLog.time >= start_dt,
        AttentionLog.time <= end_dt
    ).count()
    print(f'\nFiltered count (date range): {count}')
