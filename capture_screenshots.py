#!/usr/bin/env python3
"""
Capture screenshots of the Streamlit app using Playwright
"""
import asyncio
from playwright.async_api import async_playwright
import os
from datetime import datetime

async def capture_streamlit_screenshots():
    """Capture screenshots of the Streamlit app and any errors"""
    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        
        try:
            print("Navigating to Streamlit app...")
            # Navigate to the Streamlit app
            await page.goto("http://localhost:8501", timeout=30000)
            
            # Wait for the page to load
            await page.wait_for_load_state("networkidle", timeout=30000)
            
            # Create screenshots directory
            os.makedirs("screenshots", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Take full page screenshot
            await page.screenshot(
                path=f"screenshots/streamlit_full_page_{timestamp}.png",
                full_page=True
            )
            print(f"Full page screenshot saved: screenshots/streamlit_full_page_{timestamp}.png")
            
            # Look for error messages or warnings
            error_selectors = [
                '[data-testid="stException"]',  # Streamlit exceptions
                '.stAlert',  # Streamlit alerts
                '[role="alert"]',  # General alerts
                '.error',  # Generic error class
                '.warning',  # Warning messages
                'div:has-text("Error")',  # Divs containing "Error"
                'div:has-text("Exception")',  # Divs containing "Exception"
                'div:has-text("Warning")',  # Divs containing "Warning"
            ]
            
            error_found = False
            for i, selector in enumerate(error_selectors):
                try:
                    error_elements = await page.locator(selector).all()
                    if error_elements:
                        error_found = True
                        for j, element in enumerate(error_elements):
                            # Check if element is visible
                            if await element.is_visible():
                                # Take screenshot of the error element
                                await element.screenshot(
                                    path=f"screenshots/error_{i}_{j}_{timestamp}.png"
                                )
                                print(f"Error screenshot saved: screenshots/error_{i}_{j}_{timestamp}.png")
                                
                                # Get the error text
                                error_text = await element.inner_text()
                                print(f"Error text: {error_text}")
                except Exception as e:
                    print(f"Error checking selector {selector}: {e}")
            
            if not error_found:
                print("No visible errors found on the page")
            
            # Take a screenshot of the main content area
            try:
                main_content = page.locator('[data-testid="stAppViewContainer"]')
                if await main_content.count() > 0:
                    await main_content.screenshot(
                        path=f"screenshots/main_content_{timestamp}.png"
                    )
                    print(f"Main content screenshot saved: screenshots/main_content_{timestamp}.png")
            except Exception as e:
                print(f"Error capturing main content: {e}")
            
            # Check for any console errors
            console_messages = []
            page.on("console", lambda msg: console_messages.append(f"{msg.type}: {msg.text}"))
            
            # Wait a bit to catch any console messages
            await page.wait_for_timeout(3000)
            
            if console_messages:
                print("\nConsole messages:")
                for msg in console_messages:
                    print(f"  {msg}")
            
        except Exception as e:
            print(f"Error during screenshot capture: {e}")
            # Take emergency screenshot
            try:
                await page.screenshot(path=f"screenshots/error_emergency_{timestamp}.png")
                print(f"Emergency screenshot saved: screenshots/error_emergency_{timestamp}.png")
            except:
                pass
        
        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(capture_streamlit_screenshots())