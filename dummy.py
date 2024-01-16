import pandas as pd
import random
from datetime import datetime, timedelta

# Initialize the number of records
num_records = 10000

# Lists for possible values
assigned_teams = ['Team A', 'Team B', 'Team C', 'Team D', 'Team E', 'Team F']
resolutions = ['Resolved', 'Pending', 'Escalated', 'Closed', 'Reopened']
priorities = ['Low', 'Medium', 'High', 'Critical']
sla_times = {'Low': '1 Week', 'Medium': '3 Days', 'High': '24 Hours', 'Critical': '4 Hours'}

# Software-related customer queries, IT support responses, and their corresponding priorities
software_conversations = [
    ('Email client crashes on startup', 'Please update your email client to the latest version.', 'High'),
    ('Software installation failed with error code', 'Can you provide the specific error code you are seeing?', 'Critical'),
    ('Antivirus software not detecting malware', 'Ensure your antivirus is up to date and run a full system scan.', 'Medium'),
    ('Cannot connect to database from application', 'Check if the database service is running and accessible.', 'Critical'),
    ('Application is freezing during use', 'Try reinstalling the application and check if your system meets its requirements.', 'High'),
    ('Cannot find license key for software', 'The license key should be in your purchase confirmation email.', 'Low'),
    ('Error messages when updating a software', 'Please send the exact error messages for further diagnosis.', 'Medium'),
    ('Software X is incompatible with my OS', 'Check if there are any updates for Software X or your operating system.', 'High'),
    # More conversations can be added here
]

# Function to generate random dates for incident and resolution
def generate_dates(priority):
    start_date = datetime.now() - timedelta(days=random.randint(1, 60))
    end_date = start_date + timedelta(days=random.choice([1, 2, 3, 7, 14]))
    if priority in ['Critical', 'High']:
        end_date = start_date + timedelta(days=random.choice([1, 2, 3]))
    elif priority == 'Medium':
        end_date = start_date + timedelta(days=random.choice([3, 4, 5, 7]))
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

# Generate data
data = {
    'Incident Number': [f'INC{str(i).zfill(5)}' for i in range(num_records)],
    'Customer Query': [],
    'IT Support Response': [],
    'Assigned Team': [],
    'Resolution': [],
    'Priority': [],
    'SLA Time': [],
    'Incident Date': [],
    'Resolution Date': [],
    'Days to Resolve': []
}

for _ in range(num_records):
    query, response, priority = random.choice(software_conversations)
    assigned_team = random.choice(assigned_teams)
    resolution = random.choice(resolutions)
    sla_time = sla_times[priority]
    incident_date, resolution_date = generate_dates(priority)
    days_to_resolve = (datetime.strptime(resolution_date, '%Y-%m-%d') - datetime.strptime(incident_date, '%Y-%m-%d')).days

    data['Customer Query'].append(query)
    data['IT Support Response'].append(response)
    data['Assigned Team'].append(assigned_team)
    data['Resolution'].append(resolution)
    data['Priority'].append(priority)
    data['SLA Time'].append(sla_time)
    data['Incident Date'].append(incident_date)
    data['Resolution Date'].append(resolution_date)
    data['Days to Resolve'].append(days_to_resolve)

# Create DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file named "dummy_data.csv"
csv_file_path = 'dummy_data.csv'
df.to_csv(csv_file_path, index=False)
