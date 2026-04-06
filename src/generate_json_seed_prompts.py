import json
import random
from pathlib import Path


SEED = 42
OUT_PATH = Path("data/raw/json_prompt_seed.json")
TARGET_PER_TASK = 100


def make_json_extraction_examples():
    rows = []

    people = [
        "Sarah Chen", "Michael Torres", "Aisha Khan", "Daniel Brooks", "Emily Carter",
        "James Patel", "Olivia Reed", "Noah Bennett", "Sophia Nguyen", "Liam Foster",
        "Ava Collins", "Ethan Rivera", "Mia Stewart", "Lucas Price", "Chloe Adams",
        "Henry Cooper", "Grace Murphy", "Benjamin Ward", "Ella Hughes", "Jack Perry"
    ]

    organizations = [
        "OpenAI", "Microsoft", "Google", "NVIDIA", "Amazon",
        "Meta", "Tesla", "Apple", "Anthropic", "Databricks",
        "Adobe", "Salesforce", "Oracle", "Intel", "Cisco",
        "Palantir", "Stripe", "Snowflake", "Uber", "Airbnb"
    ]

    locations = [
        "San Francisco", "Seattle", "Austin", "Santa Clara", "New York",
        "Menlo Park", "Chicago", "Cupertino", "Boston", "Denver",
        "Dallas", "Atlanta", "Miami", "Phoenix", "Portland",
        "Los Angeles", "San Diego", "Houston", "Nashville", "Washington"
    ]

    dates = [
        "2025-03-12", "2024-11-08", "2026-01-15", "2025-07-19", "2024-09-03",
        "2025-05-21", "2026-02-10", "2025-08-14", "2026-03-01", "2024-12-18",
        "2025-01-07", "2024-10-30", "2026-04-09", "2025-06-11", "2024-08-22",
        "2025-09-05", "2026-05-16", "2024-07-29", "2025-11-13", "2026-06-20"
    ]

    order_customers = [
        "Amanda Lewis", "Robert King", "Priya Shah", "Carlos Diaz", "Rachel Green",
        "Kevin White", "Nina Patel", "Omar Hassan", "Laura Kim", "Victor Cruz",
        "Samantha Lee", "Brandon Scott", "Isabella Hall", "Mason Wright", "Zoe Baker",
        "Leo Morgan", "Hannah Kelly", "Nathan Brooks", "Ruby Foster", "Caleb Reed"
    ]

    products = [
        "wireless keyboards", "gaming monitors", "USB hubs", "laptop stands", "desk lamps",
        "office chairs", "mechanical keyboards", "noise-canceling headphones", "webcams", "portable SSDs",
        "microphones", "standing desks", "tablet chargers", "docking stations", "ethernet adapters",
        "external hard drives", "ergonomic mice", "LED monitors", "conference speakers", "phone mounts"
    ]

    states = [
        "Texas", "California", "Florida", "Illinois", "Nevada",
        "Arizona", "Georgia", "Washington", "Colorado", "Virginia",
        "Ohio", "Michigan", "Oregon", "Utah", "North Carolina",
        "South Carolina", "Pennsylvania", "New Jersey", "Massachusetts", "Tennessee"
    ]

    idx = 1

    for i in range(50):
        person = people[i % len(people)]
        org = organizations[(i * 3) % len(organizations)]
        location = locations[(i * 5) % len(locations)]
        date_ = dates[(i * 7) % len(dates)]

        rows.append({
            "id": f"json_extract_{idx:04d}",
            "task_type": "json_extraction",
            "instruction": "Extract the named entities into valid JSON with keys person, organization, location, and date.",
            "input": f"On {date_}, {person} joined {org} in {location}.",
            "expected_schema": {
                "person": "string",
                "organization": "string",
                "location": "string",
                "date": "string"
            }
        })
        idx += 1

    for i in range(50):
        customer = order_customers[i % len(order_customers)]
        product = products[(i * 2) % len(products)]
        quantity = (i % 7) + 1
        state = states[(i * 4) % len(states)]

        rows.append({
            "id": f"json_extract_{idx:04d}",
            "task_type": "json_extraction",
            "instruction": "Extract the product order details into valid JSON with keys customer_name, product, quantity, and shipping_state.",
            "input": f"{customer} ordered {quantity} {product} to be shipped to {state}.",
            "expected_schema": {
                "customer_name": "string",
                "product": "string",
                "quantity": "integer",
                "shipping_state": "string"
            }
        })
        idx += 1

    return rows


def make_schema_generation_examples():
    rows = []

    majors = [
        "Computer Science", "Data Science", "Mathematics", "Physics", "Biology",
        "Economics", "Mechanical Engineering", "Cybersecurity", "Statistics", "AI",
        "Information Systems", "Chemistry", "Electrical Engineering", "Finance", "Marketing",
        "History", "Psychology", "Political Science", "Linguistics", "Philosophy"
    ]

    reservation_events = [
        "birthday dinner", "team lunch", "anniversary meal", "client dinner", "family brunch",
        "graduation dinner", "holiday lunch", "project celebration", "networking dinner", "weekend brunch",
        "retirement party", "welcome lunch", "product launch dinner", "conference meetup", "award celebration",
        "intern farewell", "fundraiser dinner", "board lunch", "mentor meetup", "study group meal"
    ]

    idx = 1

    for i in range(50):
        major = majors[i % len(majors)]
        rows.append({
            "id": f"json_schema_{idx:04d}",
            "task_type": "schema_constrained_generation",
            "instruction": "Generate a valid JSON student profile using the required schema.",
            "input": f"Schema: name (string), age (integer), major (string), gpa (number), enrolled (boolean). Create a realistic example related to {major}.",
            "expected_schema": {
                "name": "string",
                "age": "integer",
                "major": "string",
                "gpa": "number",
                "enrolled": "boolean"
            }
        })
        idx += 1

    for i in range(50):
        event = reservation_events[i % len(reservation_events)]
        rows.append({
            "id": f"json_schema_{idx:04d}",
            "task_type": "schema_constrained_generation",
            "instruction": "Generate a valid JSON object for a restaurant reservation.",
            "input": f"Schema: customer_name (string), party_size (integer), reservation_time (string), indoor_seating (boolean). Create an example for a {event}.",
            "expected_schema": {
                "customer_name": "string",
                "party_size": "integer",
                "reservation_time": "string",
                "indoor_seating": "boolean"
            }
        })
        idx += 1

    return rows


def make_classification_examples():
    rows = []

    positive_texts = [
        "The laptop battery life is excellent and the screen is beautiful.",
        "Customer support was fast and solved my issue immediately.",
        "This keyboard feels premium and works perfectly.",
        "The update made the product much easier to use.",
        "The training session was clear, helpful, and well organized.",
        "Shipping was fast and the item arrived in perfect condition.",
        "The app now runs smoothly and the interface looks polished.",
        "The service exceeded my expectations in every way.",
        "This monitor has fantastic color accuracy and sharp resolution.",
        "The restaurant staff were friendly and the food was outstanding."
    ]

    negative_texts = [
        "The phone overheats constantly and crashes every day.",
        "The app interface is confusing and full of bugs.",
        "Shipping took too long and the item arrived damaged.",
        "The product stopped working after two days.",
        "The report contains multiple errors and missing sections.",
        "Customer support never responded to my request.",
        "The hotel room was dirty and the AC did not work.",
        "The software update made performance much worse.",
        "The billing page failed repeatedly during checkout.",
        "This chair is uncomfortable and poorly built."
    ]

    neutral_texts = [
        "The headphones are okay, nothing special but not terrible either.",
        "The hotel room was average and matched my expectations.",
        "The service was acceptable, though not especially memorable.",
        "The package arrived on time and looked as described.",
        "The class was informative but fairly standard overall.",
        "The dashboard works as expected without major issues.",
        "The meal was decent and reasonably priced.",
        "The documentation covers the basics but not much more.",
        "The product quality seems typical for this price range.",
        "The presentation was clear but not especially engaging."
    ]

    urgent_tickets = [
        "The production API is down and all customer transactions are failing.",
        "Database latency has doubled and customers are reporting timeouts.",
        "Payment webhooks are failing intermittently in production.",
        "The authentication service is offline for all users.",
        "Critical monitoring alerts show the main application is unavailable.",
        "The order processing queue is stuck and no orders are completing.",
        "Production data sync has failed and customer dashboards are blank.",
        "A security alert indicates unauthorized access to customer accounts.",
        "The checkout flow is down and revenue has stopped.",
        "The live reporting service is returning 500 errors for every request."
    ]

    high_tickets = [
        "A manager cannot log in to the dashboard before a client demo in 20 minutes.",
        "Several internal staff accounts are locked after a password policy update.",
        "The executive reporting export is broken ahead of a board meeting.",
        "A client-facing dashboard is missing key charts before a review call.",
        "A release candidate has a major bug that blocks tomorrow's launch.",
        "A partner integration is failing for an important customer segment.",
        "A field team cannot access the mobile app before an on-site event.",
        "The analytics pipeline missed today's morning refresh for leadership reports.",
        "A high-value customer reports a major workflow failure.",
        "An internal service used by multiple teams is partially unavailable."
    ]

    medium_tickets = [
        "A scheduled report has a formatting issue but still runs successfully.",
        "Users report slow loading on a non-critical settings page.",
        "One analyst cannot rename a saved filter in the admin panel.",
        "The export button is misaligned in one browser.",
        "A validation warning appears on a low-traffic admin form.",
        "Search results occasionally load slowly for archived records.",
        "The dashboard theme toggle is inconsistent across pages.",
        "A weekly email summary has minor formatting issues.",
        "A user reports duplicate tooltip text on a settings page.",
        "One chart legend overlaps slightly on smaller screens."
    ]

    low_tickets = [
        "A user wants to update their notification preferences next week.",
        "Someone requested a new color option for the dashboard theme.",
        "A team member suggested adding a shortcut to the help menu.",
        "A minor wording change was requested for the welcome email.",
        "An employee asked whether a report column could be renamed later.",
        "A suggestion was made to add more sample templates in the future.",
        "A request came in for optional profile badge customization.",
        "A user would like additional sorting choices in a later release.",
        "Someone recommended updating the FAQ wording when time permits.",
        "A request was submitted for a future dark-mode accent option."
    ]

    idx = 1

    sentiment_rows = (
        [(text, "positive") for text in positive_texts] +
        [(text, "negative") for text in negative_texts] +
        [(text, "neutral") for text in neutral_texts]
    )

    for i in range(50):
        text, label = sentiment_rows[i % len(sentiment_rows)]
        rows.append({
            "id": f"json_class_{idx:04d}",
            "task_type": "exact_label_classification",
            "instruction": "Classify the sentiment of the text and return valid JSON with keys label and rationale. Allowed labels: positive, negative, neutral.",
            "input": text,
            "expected_schema": {
                "label": "string",
                "rationale": "string"
            },
            "allowed_labels": ["positive", "negative", "neutral"],
            "target_label_hint": label
        })
        idx += 1

    priority_rows = (
        [(text, "urgent") for text in urgent_tickets] +
        [(text, "high") for text in high_tickets] +
        [(text, "medium") for text in medium_tickets] +
        [(text, "low") for text in low_tickets]
    )

    for i in range(50):
        text, label = priority_rows[i % len(priority_rows)]
        rows.append({
            "id": f"json_class_{idx:04d}",
            "task_type": "exact_label_classification",
            "instruction": "Classify the support ticket priority and return valid JSON with keys label and rationale. Allowed labels: low, medium, high, urgent.",
            "input": text,
            "expected_schema": {
                "label": "string",
                "rationale": "string"
            },
            "allowed_labels": ["low", "medium", "high", "urgent"],
            "target_label_hint": label
        })
        idx += 1

    return rows


def make_json_repair_examples():
    rows = []

    malformed = [
        (
            "{\"name\": \"Alice\", \"age\": 29, \"skills\": [\"python\", \"sql\",]}",
            {"name": "string", "age": "integer", "skills": "array[string]"}
        ),
        (
            "{name: \"Carlos\", \"department\": \"finance\", \"active\": true}",
            {"name": "string", "department": "string", "active": "boolean"}
        ),
        (
            "{\"city\": \"Austin\", \"state\": \"TX\", \"zip\": 78701,,}",
            {"city": "string", "state": "string", "zip": "integer"}
        ),
        (
            "{\"product\": \"monitor\" \"quantity\": 2, \"in_stock\": true}",
            {"product": "string", "quantity": "integer", "in_stock": "boolean"}
        ),
        (
            "{\"title\": \"Meeting\", \"participants\": [\"Ana\" \"Ben\"], \"room\": \"B12\"}",
            {"title": "string", "participants": "array[string]", "room": "string"}
        ),
        (
            "{\"course\": \"ML\", \"credits\": \"3\", \"required\": tru}",
            {"course": "string", "credits": "integer", "required": "boolean"}
        ),
        (
            "{\"employee\": \"Nina\", \"salary\": 95000 \"department\": \"IT\"}",
            {"employee": "string", "salary": "integer", "department": "string"}
        ),
        (
            "{'device': 'router', 'status': 'offline', 'priority': 'high'}",
            {"device": "string", "status": "string", "priority": "string"}
        ),
        (
            "{\"name\":\"Omar\",\"projects\":[\"A\",\"B\",],\"billable\":false}",
            {"name": "string", "projects": "array[string]", "billable": "boolean"}
        ),
        (
            "{\"date\":\"2026-04-01\",\"amount\":125.50,\"approved\":yes}",
            {"date": "string", "amount": "number", "approved": "boolean"}
        ),
        (
            "{\"user\":\"Mia\",\"roles\":[\"admin\",\"editor\",],\"enabled\":true}",
            {"user": "string", "roles": "array[string]", "enabled": "boolean"}
        ),
        (
            "{department:\"sales\", \"region\":\"west\", \"quota\":50000}",
            {"department": "string", "region": "string", "quota": "integer"}
        ),
        (
            "{\"server\":\"api-1\",\"uptime_days\":45 \"healthy\":true}",
            {"server": "string", "uptime_days": "integer", "healthy": "boolean"}
        ),
        (
            "{\"invoice_id\":\"INV-1001\",\"paid\":false,,\"amount\":850.75}",
            {"invoice_id": "string", "paid": "boolean", "amount": "number"}
        ),
        (
            "{\"book\":\"Dune\",\"authors\":[\"Frank Herbert\"],\"year\":\"1965\"}",
            {"book": "string", "authors": "array[string]", "year": "integer"}
        ),
        (
            "{\"campaign\":\"Spring\",\"budget\":12000,\"active\":tru}",
            {"campaign": "string", "budget": "integer", "active": "boolean"}
        ),
        (
            "{\"sensor\":\"T1\" \"reading\":72.4,\"unit\":\"F\"}",
            {"sensor": "string", "reading": "number", "unit": "string"}
        ),
        (
            "{'city':'Boston','temp':61,'condition':'cloudy'}",
            {"city": "string", "temp": "integer", "condition": "string"}
        ),
        (
            "{\"project\":\"Atlas\",\"milestones\":[\"design\" \"build\"],\"owner\":\"Kai\"}",
            {"project": "string", "milestones": "array[string]", "owner": "string"}
        ),
        (
            "{\"student\":\"Ella\",\"graduated\":yes,\"gpa\":3.6}",
            {"student": "string", "graduated": "boolean", "gpa": "number"}
        )
    ]

    idx = 1
    for i in range(100):
        bad_json, schema = malformed[i % len(malformed)]
        rows.append({
            "id": f"json_repair_{idx:04d}",
            "task_type": "json_repair",
            "instruction": "Repair the malformed JSON and return only corrected valid JSON.",
            "input": bad_json,
            "expected_schema": schema
        })
        idx += 1

    return rows


def make_tool_call_examples():
    rows = []

    cities = [
        ("Austin", "Texas"), ("Seattle", "Washington"), ("Miami", "Florida"), ("Denver", "Colorado"),
        ("Boston", "Massachusetts"), ("Phoenix", "Arizona"), ("Dallas", "Texas"), ("Portland", "Oregon"),
        ("Chicago", "Illinois"), ("Atlanta", "Georgia"), ("Houston", "Texas"), ("Orlando", "Florida"),
        ("San Jose", "California"), ("Raleigh", "North Carolina"), ("Salt Lake City", "Utah"),
        ("Nashville", "Tennessee"), ("Columbus", "Ohio"), ("Detroit", "Michigan"),
        ("Las Vegas", "Nevada"), ("Richmond", "Virginia")
    ]

    units = ["fahrenheit", "celsius"]

    meeting_titles = [
        "Project Sync", "Design Review", "Budget Check", "Team Standup", "Client Demo",
        "Interview Prep", "Research Review", "Weekly Planning", "Sprint Retro", "Ops Review",
        "Roadmap Update", "Data Review", "Hiring Sync", "Model Audit", "Release Check",
        "Leadership Brief", "Bug Triage", "Architecture Review", "Status Update", "Training Review"
    ]

    dates = [
        "2026-04-10", "2026-05-02", "2026-06-14", "2026-07-01", "2026-08-22",
        "2026-09-05", "2026-10-18", "2026-11-03", "2026-12-09", "2027-01-12",
        "2027-02-08", "2027-03-19", "2027-04-06", "2027-05-11", "2027-06-24",
        "2027-07-15", "2027-08-03", "2027-09-20", "2027-10-07", "2027-11-16"
    ]

    times = [
        "08:30", "09:00", "09:45", "10:15", "11:00",
        "11:30", "12:00", "13:00", "14:30", "15:00",
        "15:45", "16:15", "17:00", "17:30", "18:00",
        "08:45", "10:00", "12:15", "13:30", "16:45"
    ]

    durations = [15, 20, 25, 30, 35, 40, 45, 50, 60, 90]

    idx = 1

    for i in range(50):
        city, state = cities[i % len(cities)]
        unit = units[i % len(units)]
        unit_word = "Fahrenheit" if unit == "fahrenheit" else "Celsius"

        rows.append({
            "id": f"json_tool_{idx:04d}",
            "task_type": "tool_call_argument_generation",
            "instruction": "Generate valid JSON arguments for a weather tool call with keys city, state, and unit.",
            "input": f"Get the weather for {city}, {state} in {unit_word}.",
            "expected_schema": {
                "city": "string",
                "state": "string",
                "unit": "string"
            },
            "target_arguments_hint": {
                "city": city,
                "state": state,
                "unit": unit
            }
        })
        idx += 1

    for i in range(50):
        title = meeting_titles[i % len(meeting_titles)]
        date_ = dates[i % len(dates)]
        time_ = times[i % len(times)]
        duration = durations[i % len(durations)]

        hour, minute = map(int, time_.split(":"))
        suffix = "AM" if hour < 12 else "PM"
        display_hour = hour if 1 <= hour <= 12 else hour - 12 if hour > 12 else 12
        display_time = f"{display_hour}:{minute:02d} {suffix}"

        rows.append({
            "id": f"json_tool_{idx:04d}",
            "task_type": "tool_call_argument_generation",
            "instruction": "Generate valid JSON arguments for a calendar scheduling tool with keys title, date, time, and duration_minutes.",
            "input": f"Schedule a meeting called {title} on {date_} at {display_time} for {duration} minutes.",
            "expected_schema": {
                "title": "string",
                "date": "string",
                "time": "string",
                "duration_minutes": "integer"
            },
            "target_arguments_hint": {
                "title": title,
                "date": date_,
                "time": time_,
                "duration_minutes": duration
            }
        })
        idx += 1

    return rows


def main():
    random.seed(SEED)

    rows = []
    rows.extend(make_json_extraction_examples())
    rows.extend(make_schema_generation_examples())
    rows.extend(make_classification_examples())
    rows.extend(make_json_repair_examples())
    rows.extend(make_tool_call_examples())

    counts = {}
    for row in rows:
        counts[row["task_type"]] = counts.get(row["task_type"], 0) + 1

    print("Counts by task type before shuffle:")
    for task_type, count in sorted(counts.items()):
        print(f"  {task_type}: {count}")

    random.shuffle(rows)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(rows)} seed prompts to {OUT_PATH}")


if __name__ == "__main__":
    main()