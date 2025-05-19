import argparse
import os
from google.cloud import vision

os.environ['GOOGLE_APPLICATION_CREDENTIALS']=r''


def annotate(path: str) -> vision.WebDetection:
    """Returns web annotations given the path to an image.

    Args:
        path: path to the input image.

    Returns:
        An WebDetection object with relevant information of the
        image from the internet (i.e., the annotations).
    """
    client = vision.ImageAnnotatorClient()

    if path.startswith("http") or path.startswith("gs:"):
        image = vision.Image()
        image.source.image_uri = path

    else:
        with open(path, "rb") as image_file:
            content = image_file.read()

        image = vision.Image(content=content)

    web_detection = client.web_detection(image=image).web_detection

    return web_detection


def report(annotations: vision.WebDetection) -> None:
    """Prints detected features in the provided web annotations.

    Args:
        annotations: The web annotations (WebDetection object) from which
        the features should be parsed and printed.
    """
    if annotations.pages_with_matching_images:
        print(
            f"\n{len(annotations.pages_with_matching_images)} Pages with matching images retrieved"
        )

        for page in annotations.pages_with_matching_images:
            print(f"Url   : {page.url}")

    if annotations.full_matching_images:
        print(f"\n{len(annotations.full_matching_images)} Full Matches found: ")

        for image in annotations.full_matching_images:
            print(f"Url  : {image.url}")

    if annotations.partial_matching_images:
        print(f"\n{len(annotations.partial_matching_images)} Partial Matches found: ")

        for image in annotations.partial_matching_images:
            print(f"Url  : {image.url}")

    if annotations.web_entities:
        print(f"\n{len(annotations.web_entities)} Web entities found: ")

        for entity in annotations.web_entities:
            print(f"Score      : {entity.score}")
            print(f"Description: {entity.description}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    path_help = str(
        "The image to detect, can be web URI, "
        "Google Cloud Storage, or path to local file."
    )
    parser.add_argument("image_url", help=path_help)
    args = parser.parse_args()

    report(annotate(args.image_url))