import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# Dummy rehabilitation activity data
# ----------------------------

days = ['Nov 1', 'Nov 2', 'Nov 3', 'Nov 4', 'Nov 5', 'Nov 6', 'Nov 7']

# Sample dummy data (user preference-based metrics)
steps_per_day = [5500, 7200, 6800, 8300, 9100, 10500, 9800]          # steps
standing_balance = [12, 15, 18, 22, 24, 25, 27]                     # minutes
sit_to_stand_transitions = [5, 7, 9, 10, 12, 13, 15]                 # times per day
distance_moved = [3.2, 4.0, 3.8, 4.5, 5.1, 5.6, 5.3]                 # kilometers
mobility_score = [60, 64, 68, 72, 75, 80, 78]                        # percentage

# ----------------------------
# Create multi-line plot
# ----------------------------
plt.figure(figsize=(12, 6))

plt.plot(days, steps_per_day, marker='o', linewidth=2, label='Steps walked')
plt.plot(days, standing_balance, marker='s', linewidth=2, label='Standing balance (min)')
plt.plot(days, sit_to_stand_transitions, marker='^', linewidth=2, label='Sit-to-Stand transitions')
plt.plot(days, distance_moved, marker='D', linewidth=2, label='Distance moved (km)')
plt.plot(days, mobility_score, marker='x', linewidth=2, label='Mobility score (%)')

# ----------------------------
# Styling
# ----------------------------
plt.title('User Mobility Progress Overview ', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Activity Level', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper left', fontsize=10)
plt.tight_layout()

# ----------------------------
# Display
# ----------------------------
plt.show()
