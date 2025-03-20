import requests
import socket

# Override system DNS with Google DNS
def set_google_dns():
    import dns.resolver
    resolver = dns.resolver.Resolver()
    resolver.nameservers = ['8.8.8.8', '8.8.4.4']
    return resolver

resolver = set_google_dns()

# Test if DNS is resolving
try:
    ip_address = socket.gethostbyname('api.sportradar.com')
    print(f'api.sportradar.com resolves to {ip_address}')
except socket.gaierror as e:
    print(f'DNS resolution failed: {e}')

# Now make your API call
API_KEY = "AsO6O3efqwOVVfPu5P6ft4DtrWdHOFnZiL0nWjtu"
url = f"https://api.sportradar.com/mma/trial/v2/en/competitions.json?api_key={API_KEY}"

response = requests.get(url)
if response.status_code == 200:
    print("API call successful")
else:
    print(f"API call failed with status code: {response.status_code}")
