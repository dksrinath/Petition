from flask import render_template, flash, redirect, url_for
from datetime import datetime, timezone, timedelta
import functools
import pytz


class PetitionError(Exception):
    pass
class AuthenticationError(Exception):
    pass
def handle_errors(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except AuthenticationError as e:
            flash(str(e), "danger")
            return redirect(url_for('login'))
        except PetitionError as e:
            flash(str(e), "danger")
            return render_template('500.html'), 500
        except Exception as e:
            flash("An unexpected error occurred.", "danger")
            return render_template('500.html'), 500
    return wrapper
def get_current_utc():
    return datetime.utcnow()

def get_current_ist():
    # Add 5 hours and 30 minutes to UTC to get IST
    return datetime.utcnow() + timedelta(hours=5, minutes=30)

def format_datetime(dt):
    """Format datetime in YYYY-MM-DD HH:MM:SS format"""
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime('%Y-%m-%d %H:%M:%S')

def format_datetime_display(dt):
    """Format datetime for display with UTC indicator"""
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return f"{dt.strftime('%Y-%m-%d %H:%M:%S')} UTC"