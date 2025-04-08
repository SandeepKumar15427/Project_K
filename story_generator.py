import os
import re
import time
import json
import google.generativeai as genai
from typing import List, Dict, Tuple, Any
import logging  # Added for better error reporting

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Configure Gemini (only needs to be done once)
try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    logging.error(f"Failed to configure Gemini: {e}")
    raise

GEMINI_MODEL_NAME = "gemini-1.5-pro-latest"
MAX_TOKENS_PER_CHUNK = 30000  # Max tokens for chapter generation output
CONTEXT_WINDOW_SIZE = 4000   # Characters to retain from the end of previous chunks
ENTITY_EXTRACTION_TEXT_LIMIT = 8000 # Characters to feed into entity extraction
API_RETRY_DELAY = 5  # Seconds to wait before retrying API calls
API_MAX_RETRIES = 3  # Maximum number of retries for API calls

# Initialize Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Gemini Model
try:
    model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    logging.info(f"Gemini model '{GEMINI_MODEL_NAME}' initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize Gemini model: {e}")
    raise

# --- Utility Function for Robust API Calls ---
def generate_with_retry(prompt: str, **kwargs) -> str:
    """Generates content using the Gemini model with retry logic."""
    retries = 0
    while retries < API_MAX_RETRIES:
        try:
            response = model.generate_content(prompt, **kwargs)
            # Handle potential safety blocks or empty responses
            if not response.parts:
                 # Check if it was blocked due to safety settings
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    logging.warning(f"API call blocked due to safety reasons: {response.prompt_feedback.block_reason}")
                    # Depending on the reason, you might want to retry or raise an error
                    # For now, let's treat it as a failure for retry purposes
                else:
                    logging.warning("API call returned empty response.")
                # Consider raising an error or returning a default value if retries fail
                raise genai.types.BlockedPromptException("Blocked or empty response")


            return response.text
        except (genai.types.BlockedPromptException, genai.types.StopCandidateException) as safety_err:
             logging.warning(f"Content generation potentially blocked: {safety_err}. Retrying...")
             retries += 1
             time.sleep(API_RETRY_DELAY * (retries + 1)) # Exponential backoff might be better
        except Exception as e:
            logging.warning(f"API call failed: {e}. Retrying ({retries + 1}/{API_MAX_RETRIES})...")
            retries += 1
            time.sleep(API_RETRY_DELAY * (retries + 1)) # Exponential backoff might be better

    logging.error(f"API call failed after {API_MAX_RETRIES} retries.")
    # Decide how to handle permanent failure: raise error or return default/empty string
    raise RuntimeError(f"Failed to generate content after {API_MAX_RETRIES} retries.")


# --- Context Management System ---
class StoryContext:
    """Manages the story's context, including characters, locations, plot points, and recent text."""
    def __init__(self):
        self.master_outline: Dict[str, Any] = {} # Changed to Dict assuming structure later
        self.character_bible: Dict[str, Dict[str, Any]] = {}
        self.location_guide: Dict[str, str] = {}
        self.plot_points: List[str] = []
        self.previous_chunks_context: List[str] = [] # Renamed for clarity
        self.current_status: Dict[str, Any] = {} # Consider defining its structure more

    def update_context(self, new_content: str) -> None:
        """Extract and store key context elements from newly generated text."""
        logging.info("Updating context from new content...")
        # --- Extract Entities using Gemini ---
        extracted_data = self._extract_entities_from_text(new_content)

        # --- Update Characters ---
        if "characters" in extracted_data and isinstance(extracted_data["characters"], dict):
            characters = extracted_data["characters"]
            for char_name, details in characters.items():
                if isinstance(details, dict): # Ensure details are a dictionary
                    if char_name not in self.character_bible:
                        self.character_bible[char_name] = details
                        logging.info(f"Added new character: {char_name}")
                    else:
                        # Simple update, could be more sophisticated (e.g., merging lists)
                        self.character_bible[char_name].update(details)
                        logging.info(f"Updated character: {char_name}")
                else:
                     logging.warning(f"Invalid details format for character '{char_name}': {details}")
        else:
            logging.warning("No valid 'characters' dictionary found in extracted data.")


        # --- Update Locations ---
        if "locations" in extracted_data and isinstance(extracted_data["locations"], dict):
            locations = extracted_data["locations"]
            # Simple update, overwrites existing description if name clashes
            self.location_guide.update(locations)
            for loc_name in locations:
                 logging.info(f"Updated/Added location: {loc_name}")
        else:
            logging.warning("No valid 'locations' dictionary found in extracted data.")


        # --- Track Plot Points ---
        if "plot_points" in extracted_data and isinstance(extracted_data["plot_points"], list):
            new_plot_points = extracted_data["plot_points"]
            # Avoid adding duplicates if desired, or just append all
            self.plot_points.extend(p for p in new_plot_points if p and isinstance(p, str)) # Add only non-empty strings
            logging.info(f"Added {len(new_plot_points)} potential plot points.")
        else:
            logging.warning("No valid 'plot_points' list found in extracted data.")


        # --- Maintain Rolling Context Window ---
        # Keep the end portion of the latest chunk
        context_snippet = new_content[-CONTEXT_WINDOW_SIZE:]
        # Add it to the beginning of the list, keep only the last 3 snippets
        self.previous_chunks_context.insert(0, context_snippet)
        self.previous_chunks_context = self.previous_chunks_context[:3] # Keep last 3
        logging.info("Updated previous chunks context.")

    def get_context_summary(self) -> str:
        """Generate a condensed context summary string for prompts."""
        # Join the recent snippets, most recent first
        recent_context = "\n...\n".join(self.previous_chunks_context)

        # Use try-except for JSON serialization as complex objects might cause issues
        try:
            char_json = json.dumps(self.character_bible, indent=2)
        except TypeError:
            char_json = str(self.character_bible) # Fallback to string representation
            logging.warning("Could not serialize character_bible to JSON.")
        try:
            loc_json = json.dumps(self.location_guide, indent=2)
        except TypeError:
            loc_json = str(self.location_guide) # Fallback
            logging.warning("Could not serialize location_guide to JSON.")

        summary_parts = [
            f"### Characters:\n{char_json}",
            f"### Locations:\n{loc_json}",
            f"### Active Plot Points:\n{self.plot_points}", # Assuming plot points are simple strings
            f"### Recent Context Snippets (Most Recent First):\n{recent_context}"
        ]
        return "\n\n".join(summary_parts)

    def _extract_entities_from_text(self, text: str) -> Dict[str, Any]:
        """Use Gemini to extract structured entities (characters, locations, plot points) from text."""
        logging.info("Attempting to extract entities...")
        # Use a representative chunk of text for extraction
        text_chunk = text[:ENTITY_EXTRACTION_TEXT_LIMIT]

        # Unified prompt asking for all relevant entities in one go
        prompt = f"""Analyze the following story segment and extract key entities.
Respond ONLY with a valid JSON object containing 'characters', 'locations', and 'plot_points' keys.
Format:
{{
    "characters": {{
        "Character Name 1": {{
            "description": "Brief description or update.",
            "traits": ["Trait1", "Trait2"],
            "current_status": "Current situation or emotional state."
        }},
        "Character Name 2": {{ ... }}
    }},
    "locations": {{
        "Location Name 1": "Description or recent events at this location.",
        "Location Name 2": "..."
    }},
    "plot_points": [
        "Summary of a key event or development.",
        "Another significant plot advancement."
    ]
}}

Story Segment:
\"\"\"
{text_chunk}
\"\"\"

JSON Response:
"""
        try:
            # Use the robust generation function
            response_text = generate_with_retry(prompt)

            # Clean potential markdown code fences or other artifacts
            cleaned_response = re.sub(r"^```json\s*|\s*```$", "", response_text.strip(), flags=re.MULTILINE)

            # Attempt to parse the JSON
            extracted_data = json.loads(cleaned_response)
            logging.info("Successfully extracted entities.")
            # Basic validation (check if it's a dict)
            if not isinstance(extracted_data, dict):
                 logging.warning(f"Entity extraction returned non-dict type: {type(extracted_data)}")
                 return {}
            return extracted_data

        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode JSON from entity extraction response: {e}")
            logging.debug(f"Raw response was: {response_text}") # Log raw response for debugging
            return {} # Return empty dict on JSON error
        except (RuntimeError, Exception) as e: # Catch retry failures or other unexpected errors
             logging.error(f"Failed to extract entities due to API or other error: {e}")
             return {} # Return empty dict on other errors

# --- Chunked Generation System ---
class StoryGenerator:
    """Handles the generation of the story outline and individual chapters."""
    def __init__(self):
        self.context = StoryContext()
        self.chapter_files: List[str] = []

    def generate_master_outline(self, premise: str) -> str:
        """Creates a detailed, hierarchical story outline using the Gemini model."""
        logging.info("Generating master outline...")
        prompt = f"""Create a detailed, hierarchical master outline for a long-form story based on the following premise.
The outline should include:
1.  **Overall Structure:** A clear 3-act structure (Setup, Confrontation, Resolution) with key turning points.
2.  **Chapter Breakdown:** A list of proposed chapters, each with a concise summary (1-3 sentences) of its main events and purpose. Number the chapters sequentially.
3.  **Character Arcs:** Brief descriptions of the main characters and their intended development/arcs throughout the story.
4.  **Key Plot Points:** A list of major plot events, mysteries, or conflicts that will drive the narrative.
5.  **Major Locations:** Descriptions of the primary settings where the story takes place.

Premise:
\"\"\"
{premise}
\"\"\"

Respond in well-structured Markdown format with clear section headers (e.g., using ## or ###).
"""
        try:
            # Use the robust generation function
            outline_content = generate_with_retry(prompt)
            # Storing the raw Markdown text. Could potentially parse it into self.context.master_outline dict later.
            self.context.master_outline = {"raw_markdown": outline_content} # Store as dict entry
            logging.info("Master outline generated successfully.")
            return outline_content
        except RuntimeError as e:
             logging.error(f"Failed to generate master outline: {e}")
             # Return empty string or re-raise depending on desired behavior
             return ""


    def generate_chapter(self, chapter_num: int) -> str:
        """Generates a single chapter, incorporating context and adhering to the outline."""
        logging.info(f"Generating Chapter {chapter_num}...")
        context_summary = self.context.get_context_summary()
        master_outline_text = self.context.master_outline.get("raw_markdown", "No outline available.")

        prompt = f"""You are an expert storyteller continuing a long-form narrative. Write Chapter {chapter_num} of this story.

Follow these instructions carefully:
1.  **Consistency:** Maintain strict consistency with the characters, events, tone, and established facts from the provided Context Summary and Master Outline.
2.  **Character Development:** Advance the arcs of relevant characters as suggested by the outline and previous events. Show, don't just tell, their internal states and changes.
3.  **Plot Progression:** Move the plot forward according to the Master Outline's plan for this chapter. Introduce necessary events, conflicts, or revelations.
4.  **Use Context:** Seamlessly integrate information from the Character Bible, Location Guide, and Recent Context Snippets. Refer to established locations and character details naturally.
5.  **Pacing and Flow:** Ensure the chapter has a good pace, building tension or emotion appropriately. End the chapter logically, perhaps with a hook or a smooth transition point for the next chapter.
6.  **Word Count:** Aim for approximately 2000-2500 words for this chapter.
7.  **Format:** Start the chapter *exactly* with 'Chapter {chapter_num}: [Your Chapter Title]' on the first line, followed by the narrative.

### Context Summary:
{context_summary}

### Master Outline:
{master_outline_text}

---
Begin Chapter {chapter_num}:
"""
        try:
            # Use robust generation with specified token limit
            chapter_content = generate_with_retry(
                prompt,
                generation_config={"max_output_tokens": MAX_TOKENS_PER_CHUNK}
            )

            # Basic check if content seems valid
            if not chapter_content or not chapter_content.strip():
                 logging.error(f"Chapter {chapter_num} generation resulted in empty content.")
                 return "" # Indicate failure

            # Update context *after* successful generation
            self.context.update_context(chapter_content)

            # Save the chapter
            filename = self._save_chapter(chapter_content, chapter_num)
            logging.info(f"Chapter {chapter_num} generated and saved to {filename}.")
            return filename
        except RuntimeError as e:
            logging.error(f"Failed to generate Chapter {chapter_num}: {e}")
            return "" # Indicate failure

    def _save_chapter(self, content: str, number: int) -> str:
        """Saves the generated chapter content to a text file with metadata."""
        # Create a unique filename including a timestamp
        timestamp = int(time.time())
        filename = f"chapter_{number:03d}_{timestamp}.txt" # Pad chapter number for sorting

        metadata_comment = f"\n\n--- METADATA (GENERATION CONTEXT) ---\n"
        # Attempt to serialize context parts individually for robustness
        try:
            context_dump = {
                "generation_timestamp": timestamp,
                "chapter_number": number,
                # Only include serializable parts, avoid dumping the whole potentially huge context object
                "character_bible_snapshot": self.context.character_bible,
                "location_guide_snapshot": self.context.location_guide,
                "plot_points_at_generation": self.context.plot_points,
                 # Consider omitting previous_chunks_context if too large or redundant
                # "previous_chunks_context_snapshot": self.context.previous_chunks_context,
                # Avoid dumping the raw outline in every file unless necessary
                # "master_outline_snapshot": self.context.master_outline
            }
            metadata_json = json.dumps(context_dump, indent=2)
        except Exception as e:
            logging.error(f"Could not serialize metadata for {filename}: {e}")
            metadata_json = f"Error serializing metadata: {e}"

        metadata_to_write = metadata_comment + metadata_json

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
                # Append metadata as a distinct block at the end
                f.write(metadata_to_write)
            self.chapter_files.append(filename)
            logging.info(f"Successfully saved chapter {number} to {filename}")
            return filename
        except IOError as e:
            logging.error(f"Failed to write chapter file {filename}: {e}")
            return "" # Indicate failure

    def assemble_story(self, output_file: str = "complete_story.txt") -> str:
        """Combines generated chapter files into a single final manuscript."""
        logging.info(f"Assembling story from {len(self.chapter_files)} chapters...")
        full_text_parts = []

        # Sort files, assuming filenames like chapter_001_ts.txt, chapter_002_ts.txt etc.
        sorted_files = sorted(self.chapter_files)

        for i, fname in enumerate(sorted_files):
            try:
                with open(fname, 'r', encoding='utf-8') as f:
                    # Read the whole file content
                    file_content = f.read()
                    # Split based on the metadata separator
                    content_parts = file_content.split("\n\n--- METADATA (GENERATION CONTEXT) ---", 1)
                    # Keep only the part *before* the separator
                    chapter_text = content_parts[0].strip()
                    if chapter_text: # Ensure we don't add empty strings
                        full_text_parts.append(chapter_text)
                    else:
                        logging.warning(f"Chapter file {fname} appeared to have no content before metadata.")
            except FileNotFoundError:
                logging.error(f"Chapter file not found during assembly: {fname}")
            except IOError as e:
                logging.error(f"Error reading chapter file {fname}: {e}")
            except Exception as e:
                 logging.error(f"Unexpected error processing file {fname}: {e}")


        if not full_text_parts:
             logging.error("No chapter content found to assemble.")
             return ""


        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                # Join chapters with double newlines for separation
                f.write("\n\n".join(full_text_parts))
            logging.info(f"Complete story assembled and saved to {output_file}")
            return output_file
        except IOError as e:
            logging.error(f"Failed to write final story file {output_file}: {e}")
            return "" # Indicate failure

# --- Main Workflow ---
if __name__ == "__main__":
    generator = StoryGenerator()

    premise = input("Enter the story premise: ")
    if not premise.strip():
        print("Premise cannot be empty. Exiting.")
        exit()

    print("\nGenerating master outline...")
    outline = generator.generate_master_outline(premise)
    if not outline:
         print("Failed to generate outline. Exiting.")
         exit()

    # Save the outline for reference
    try:
        with open("master_outline.md", "w", encoding="utf-8") as f:
            f.write(outline)
        print("Master outline saved to master_outline.md")
    except IOError as e:
        print(f"Warning: Could not save outline file: {e}")

    print(f"\nMaster Outline Generated ({len(outline.split())} words approx):\n---")
    # Print a snippet of the outline
    print(outline[:1000] + "..." if len(outline) > 1000 else outline)
    print("---\n")

    while True:
        try:
            num_chapters_input = input("How many chapters do you want to generate? ")
            num_chapters = int(num_chapters_input)
            if num_chapters > 0:
                break
            else:
                print("Please enter a positive number of chapters.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    start_time = time.time()
    chapters_generated_count = 0
    for chapter_num in range(1, num_chapters + 1):
        print(f"\n--- Generating Chapter {chapter_num}/{num_chapters} ---")
        try:
            filename = generator.generate_chapter(chapter_num)
            if filename: # Check if generation and saving succeeded
                print(f"Chapter {chapter_num} generated successfully. Saved to {filename}")
                chapters_generated_count += 1
            else:
                print(f"Skipping Chapter {chapter_num} due to generation failure.")
                # Optionally break or ask user to continue

            # Rate limit protection - consider making this configurable or smarter
            print(f"Waiting {API_RETRY_DELAY} seconds before next chapter...")
            time.sleep(API_RETRY_DELAY)
        except Exception as e:
             # Catch unexpected errors during the loop
             logging.error(f"An unexpected error occurred during chapter {chapter_num} generation: {e}")
             print(f"An error occurred for chapter {chapter_num}, check logs. Attempting to continue...")
             time.sleep(API_RETRY_DELAY) # Wait even after failure

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n--- Generation Summary ---")
    print(f"Successfully generated {chapters_generated_count}/{num_chapters} chapters.")
    print(f"Total generation time: {total_time:.2f} seconds.")

    if chapters_generated_count > 0:
        print("\nAssembling the final story...")
        final_file = generator.assemble_story()
        if final_file:
             print(f"\nStory generation complete! Final manuscript saved to: {final_file}")
        else:
             print("\nStory assembly failed. Check logs for details. Individual chapter files may still exist.")
    else:
        print("\nNo chapters were successfully generated. Story assembly skipped.")


# --- Additional Features (Defined but not called in main workflow by default) ---

def consistency_check(generator: StoryGenerator) -> str:
    """Analyzes generated content for consistency errors using the LLM."""
    logging.info("Performing consistency check...")
    if not generator.context.previous_chunks_context:
        logging.warning("No content generated yet to perform consistency check.")
        return "No content available for consistency check."

    # Use a more focused summary for the check if needed, or the standard one
    context_summary = generator.context.get_context_summary()

    prompt = f"""Analyze the provided story context summary for potential continuity and consistency errors.
Focus specifically on:
1.  **Character Inconsistencies:** Contradictory actions, motivations, descriptions, or knowledge between different points in the story.
2.  **Plot Contradictions:** Events that conflict with previously established facts or timelines. Unresolved plot threads that seem forgotten.
3.  **Timeline Errors:** Confusing or impossible sequences of events. Inconsistent passage of time.
4.  **Location Mismatches:** Descriptions or uses of locations that contradict earlier information.

Context Summary:
{context_summary}

---
Provide a detailed report identifying specific potential inconsistencies. If none are found, state that clearly.
Report:
"""
    try:
        report = generate_with_retry(prompt)
        report_filename = "consistency_report.txt"
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report)
        logging.info(f"Consistency report generated and saved to {report_filename}")
        return report
    except RuntimeError as e:
        logging.error(f"Failed to generate consistency report: {e}")
        return f"Error generating consistency report: {e}"
    except IOError as e:
         logging.error(f"Failed to save consistency report: {e}")
         # Return the report content even if saving failed
         return report if 'report' in locals() else f"Error generating consistency report and failed to save: {e}"


def rewrite_chapter(filename: str, notes: str, generator: StoryGenerator) -> str:
    """Regenerates a specific chapter based on provided notes, attempting to maintain context."""
    logging.info(f"Attempting to rewrite chapter file: {filename} with notes: {notes[:100]}...")
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            original_full_content = f.read()
            # Extract original content before metadata
            content_parts = original_full_content.split("\n\n--- METADATA (GENERATION CONTEXT) ---", 1)
            original_content = content_parts[0].strip()
            # Extract original metadata if needed for context, though the generator's *current* context is likely more relevant
            # original_metadata_str = content_parts[1] if len(content_parts) > 1 else ""

    except FileNotFoundError:
        logging.error(f"Cannot rewrite chapter: File not found - {filename}")
        return ""
    except IOError as e:
        logging.error(f"Error reading chapter file {filename} for rewrite: {e}")
        return ""

    # Use the generator's *current* context summary for rewriting
    context_summary = generator.context.get_context_summary()
    master_outline_text = generator.context.master_outline.get("raw_markdown", "No outline available.")

    # Extract chapter number from filename if possible (requires consistent naming)
    match = re.search(r'chapter_(\d+)_', filename)
    chapter_num_str = f"Chapter {match.group(1)}" if match else "the specified chapter"


    prompt = f"""Rewrite {chapter_num_str} based on the following notes.

**Rewrite Notes:**
\"\"\"
{notes}
\"\"\"

**Original Chapter Content:**
\"\"\"
{original_content}
\"\"\"

**Task:**
- Incorporate the changes requested in the notes.
- Maintain consistency with the overall story context provided below.
- Preserve the core events and purpose of the chapter unless the notes specify otherwise.
- Ensure the rewritten chapter flows naturally and fits within the narrative.
- Format the output starting *exactly* with '{chapter_num_str}: [New or Original Chapter Title]'.

**Current Story Context Summary:**
{context_summary}

**Master Outline:**
{master_outline_text}

---
Begin Rewritten {chapter_num_str}:
"""

    try:
        rewritten_content = generate_with_retry(
            prompt,
            generation_config={"max_output_tokens": MAX_TOKENS_PER_CHUNK} # Use same limit as original generation
        )

        # Overwrite the original file with the rewritten content + NEW metadata
        # (Consider backing up the original first)
        logging.warning(f"Overwriting original file {filename} with rewritten content.")
        # Use the same saving logic, which appends current context metadata
        new_filename = generator._save_chapter(rewritten_content, int(match.group(1)) if match else 0) # Reuse save logic
        if new_filename == filename: # _save_chapter might generate a new timestamped name
            logging.info(f"Chapter successfully rewritten and saved to {filename}")
            # Important: Update the main context based on the *rewritten* content
            generator.context.update_context(rewritten_content)
            logging.info("Generator context updated with rewritten chapter content.")
            return filename
        else:
             logging.warning(f"Rewritten chapter saved to NEW file {new_filename} instead of overwriting {filename}. Manual cleanup might be needed.")
             return new_filename


    except RuntimeError as e:
        logging.error(f"Failed to rewrite chapter {filename}: {e}")
        return "" # Indicate failure
    except Exception as e:
         logging.error(f"An unexpected error occurred during rewrite of {filename}: {e}")
         return ""

