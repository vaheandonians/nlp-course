import Levenshtein
from enum import Enum


class Scale(Enum):
    NOT_APPLICABLE = "Not Applicable"
    AS_IS = "AS IS"
    THOUSAND = "THOUSAND"
    MILLION = "MILLION"
    BILLION = "BILLION"

    @staticmethod
    def get_options_in_natural_language() -> str:
        return (f"The scale can be either "
                f"{Scale.NOT_APPLICABLE.value}, "
                f"{Scale.AS_IS.value}, "
                f"{Scale.THOUSAND.value}, "
                f"{Scale.MILLION.value}, or "
                f"{Scale.BILLION.value}.")

    @staticmethod
    def match(value: str):
        value = value.upper()
        for scale in Scale:
            if scale.value == value:
                return scale.value
            if Levenshtein.distance(scale.value, value) == 1:
                return scale.value
        return None


currency_codes = [
    "Not Applicable",
    "USD", "EUR", "JPY", "GBP", "AUD", "CAD", "CHF", "CNY", "SEK", "NZD",
    "MXN", "SGD", "HKD", "NOK", "KRW", "TRY", "RUB", "INR", "BRL", "ZAR",
    "DKK", "PLN", "THB", "IDR", "HUF", "CZK", "ILS", "CLP", "PHP", "AED",
    "COP", "SAR", "MYR", "RON", "AMD", "AFN", "ALL", "AOA", "ARS", "AWG",
    "AZN", "BAM", "BBD", "BDT", "BGN", "BHD", "BIF", "BMD", "BND", "BOB",
    "BSD", "BTN", "BWP", "BYN", "BZD", "CDF", "CRC", "CUP", "CVE", "DJF",
    "DOP", "DZD", "EGP", "ERN", "ETB", "FJD", "FKP", "FOK", "GEL", "GGP",
    "GHS", "GIP", "GMD", "GNF", "GTQ", "GYD", "HNL", "HTG", "IQD", "IRR",
    "ISK", "JEP", "JMD", "JOD", "KES", "KGS", "KHR", "KID", "KMF", "KPW",
    "KWD", "KYD", "KZT", "LAK", "LBP", "LKR", "LRD", "LSL", "LYD", "MAD",
    "MDL", "MGA", "MKD", "MMK", "MNT", "MOP", "MRU", "MUR", "MVR", "MWK",
    "MZN", "NAD", "NGN", "NIO", "NPR", "OMR", "PAB", "PEN", "PGK", "PKR",
    "PYG", "QAR", "RSD", "RWF", "SBD", "SCR", "SDG", "SHP", "SLE", "SLL",
    "SOS", "SRD", "SSP", "STN", "SYP", "SZL", "TJS", "TMT", "TND", "TOP",
    "TTD", "TVD", "TZS", "UGX", "UYU", "UZS", "VES", "VND", "VUV", "WST",
    "XAF", "XCD", "XOF", "XPF", "YER", "ZMW", "ZWL"
]
