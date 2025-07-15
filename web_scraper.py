# scrape_one_course.py
import json
import os
from playwright.sync_api import sync_playwright
import re

def scrape_one_course():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # headless=False to see what's happening
        page = browser.new_page()
        page.goto("https://www.cmucourses.com")
        try:
            page.click("button:has-text('Continue without logging in')", timeout=5000)
            print("Clicked 'Continue without logging in'")
        except Exception as e:
            print("No login popup to dismiss:", e)

        # Open the department dropdown and select 'Computer Science'
        try:
            # Click the department dropdown to open it
            page.click("[id^='headlessui-combobox-button']")
            # Click the 'Computer Science' option
            page.click("li:has-text('Computer Science')")
            print("Selected Computer Science department from dropdown")
        except Exception as e:
            print("Could not select Computer Science department from dropdown:", e)
        page.wait_for_load_state("networkidle")
        
        print(page.content())  # For debugging: see what the page looks like

        try:
            page.wait_for_selector("div.bg-white.border-gray-100.rounded.border.p-6", timeout=60000)
        except Exception as e:
            print("Selector not found:", e)
            browser.close()
            return

        while True:
            # Get all course cards on the current page
            cards = page.query_selector_all("div.bg-white.border-gray-100.rounded.border.p-6")
            if not cards:
                print("No course cards found on this page.")
                break

            for card in cards:
                # Extract course number from the first <span> inside the title div
                course_num_el = card.query_selector("div.text-lg.text-gray-800 > span")
                course_number = course_num_el.inner_text().strip() if course_num_el else "unknown_course"

                title_el = card.query_selector("div.text-lg.text-gray-800")
                title = title_el.inner_text().strip() if title_el else "N/A"
                units_el = card.query_selector("div.text-lg.text-gray-700")
                units = units_el.inner_text().strip() if units_el else "N/A"
                desc_el = card.query_selector("div.text-sm.leading-relaxed.text-gray-600")
                description = desc_el.inner_text().strip() if desc_el else "N/A"
                prereq_el = card.query_selector(".font-semibold:text('Prereq') ~ .text-md.text-gray-500")
                prereq = prereq_el.inner_text().strip() if prereq_el else "None"
                coreq_el = card.query_selector(".font-semibold:text('Coreq') ~ .text-md.text-gray-500")
                coreq = coreq_el.inner_text().strip() if coreq_el else "None"

                os.makedirs("documents", exist_ok=True)
                file_path = f"documents/{course_number}.txt"
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(f"Title: {title}\n")
                    f.write(f"Course Number: {course_number}\n")
                    f.write(f"Units: {units}\n")
                    f.write(f"Prerequisites: {prereq}\n")
                    f.write(f"Corequisites: {coreq}\n")
                    f.write(f"Description: {description}\n")
                print(f"Course scraped and saved successfully as {file_path}.")

            # Select the "Next" button based on its position
            pagination_buttons = page.locator("div.justify-center button")
            if pagination_buttons.count() >= 2:
                next_button = pagination_buttons.nth(1)
                if next_button.is_enabled():
                    next_button.scroll_into_view_if_needed()
                    next_button.click()
                    page.wait_for_load_state("networkidle")
                else:
                    print("Next button is disabled. Stopping.")
                    break
            else:
                print("Next button not found. Stopping.")
                break
        browser.close()

if __name__ == "__main__":
    scrape_one_course()
