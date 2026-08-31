from pathlib import Path
import xml.etree.ElementTree as ET


class OFACParser:
    """
    Simple parser for the OFAC SDN XML file.

    Extracts only the fields needed for our KYC demo:
        - uid
        - first name
        - last name
        - aliases
        - type
        - date of birth
        - nationality
        - citizenship
        - countries
        - sanctions programs
    """

    def __init__(self, xml_path: str):
        self.xml_path = Path(xml_path)

        if not self.xml_path.exists():
            raise FileNotFoundError(
                f"OFAC XML file not found: {self.xml_path}"
            )

    def parse(self) -> list[dict]:
        """
        Parse the OFAC XML file.

        Returns:
            List of normalized OFAC records.
        """

        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        # -----------------------------------------------------
        # OFAC XML uses a default namespace.
        # -----------------------------------------------------

        namespace = {
            "ofac": "https://sanctionslistservice.ofac.treas.gov/api/PublicationPreview/exports/XML"
        }

        records = []

        for entry in root.findall("ofac:sdnEntry", namespace):

            uid = self._get_text(
                entry,
                "ofac:uid",
                namespace
            )

            first_name = self._get_text(
                entry,
                "ofac:firstName",
                namespace
            )

            last_name = self._get_text(
                entry,
                "ofac:lastName",
                namespace
            )

            sdn_type = self._get_text(
                entry,
                "ofac:sdnType",
                namespace
            )

            # -------------------------------------------------
            # Build primary name
            # -------------------------------------------------

            name_parts = []

            if first_name:
                name_parts.append(first_name)

            if last_name:
                name_parts.append(last_name)

            name = " ".join(name_parts).strip()

            # -------------------------------------------------
            # Aliases
            # -------------------------------------------------

            aliases = []

            aka_list = entry.find(
                "ofac:akaList",
                namespace
            )

            if aka_list is not None:

                for aka in aka_list.findall(
                    "ofac:aka",
                    namespace
                ):

                    aka_first = self._get_text(
                        aka,
                        "ofac:firstName",
                        namespace
                    )

                    aka_last = self._get_text(
                        aka,
                        "ofac:lastName",
                        namespace
                    )

                    alias_parts = []

                    if aka_first:
                        alias_parts.append(aka_first)

                    if aka_last:
                        alias_parts.append(aka_last)

                    alias = " ".join(
                        alias_parts
                    ).strip()

                    if alias:
                        aliases.append(alias)

            # -------------------------------------------------
            # Date of birth
            # -------------------------------------------------

            dates_of_birth = []

            dob_list = entry.find(
                "ofac:dateOfBirthList",
                namespace
            )

            if dob_list is not None:

                for dob_item in dob_list.findall(
                    "ofac:dateOfBirthItem",
                    namespace
                ):

                    dob = self._get_text(
                        dob_item,
                        "ofac:dateOfBirth",
                        namespace
                    )

                    if dob:
                        dates_of_birth.append(dob)

            # -------------------------------------------------
            # Nationality
            # -------------------------------------------------

            nationalities = []

            nationality_list = entry.find(
                "ofac:nationalityList",
                namespace
            )

            if nationality_list is not None:

                for nationality in nationality_list.findall(
                    "ofac:nationality",
                    namespace
                ):

                    country = self._get_text(
                        nationality,
                        "ofac:country",
                        namespace
                    )

                    if country:
                        nationalities.append(country)

            # -------------------------------------------------
            # Citizenship
            # -------------------------------------------------

            citizenships = []

            citizenship_list = entry.find(
                "ofac:citizenshipList",
                namespace
            )

            if citizenship_list is not None:

                for citizenship in citizenship_list.findall(
                    "ofac:citizenship",
                    namespace
                ):

                    country = self._get_text(
                        citizenship,
                        "ofac:country",
                        namespace
                    )

                    if country:
                        citizenships.append(country)

            # -------------------------------------------------
            # Countries from addresses
            # -------------------------------------------------

            countries = []

            address_list = entry.find(
                "ofac:addressList",
                namespace
            )

            if address_list is not None:

                for address in address_list.findall(
                    "ofac:address",
                    namespace
                ):

                    country = self._get_text(
                        address,
                        "ofac:country",
                        namespace
                    )

                    if country:
                        countries.append(country)

            # -------------------------------------------------
            # Sanctions programs
            # -------------------------------------------------

            programs = []

            program_list = entry.find(
                "ofac:programList",
                namespace
            )

            if program_list is not None:

                for program in program_list.findall(
                    "ofac:program",
                    namespace
                ):

                    if program.text:
                        programs.append(
                            program.text.strip()
                        )

            # -------------------------------------------------
            # Store normalized record
            # -------------------------------------------------

            records.append(
                {
                    "uid": uid,
                    "name": name,
                    "first_name": first_name,
                    "last_name": last_name,
                    "aliases": aliases,
                    "sdn_type": sdn_type,
                    "dates_of_birth": dates_of_birth,
                    "nationalities": nationalities,
                    "citizenships": citizenships,
                    "countries": countries,
                    "programs": programs,
                }
            )

        return records

    @staticmethod
    def _get_text(
        element,
        path: str,
        namespace: dict
    ):
        """Safely extract text from an XML element."""

        child = element.find(
            path,
            namespace
        )

        if child is None:
            return None

        if child.text is None:
            return None

        return child.text.strip()


# ---------------------------------------------------------
# Simple test
# ---------------------------------------------------------

if __name__ == "__main__":

    xml_path = (
        "data/sanctions/ofac_sdn.xml"
    )

    parser = OFACParser(xml_path)

    records = parser.parse()

    print(
        f"OFAC records loaded: {len(records)}"
    )

    # Display first five records
    for record in records[:5]:

        print("\n----------------------------")

        print(
            f"UID: {record['uid']}"
        )

        print(
            f"Name: {record['name']}"
        )

        print(
            f"Type: {record['sdn_type']}"
        )

        print(
            f"Aliases: {record['aliases']}"
        )

        print(
            f"DOB: {record['dates_of_birth']}"
        )

        print(
            f"Countries: {record['countries']}"
        )

        print(
            f"Programs: {record['programs']}"
        )