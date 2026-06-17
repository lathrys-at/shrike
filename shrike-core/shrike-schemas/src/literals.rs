//! Literal-valued field types — Pydantic's `Literal[True]` / `Literal["reloaded"]`
//! for fields that are *not* a union's tag (those are enum variants instead).
//!
//! Each type (de)serializes exactly its one value and advertises a `const`
//! JSON Schema, so the wire shape and the schema both match the Pydantic side.

use std::borrow::Cow;

use schemars::{JsonSchema, Schema, SchemaGenerator};
use serde::de::Error as DeError;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

macro_rules! literal_bool {
    ($name:ident, $value:literal) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
        pub struct $name;

        impl Serialize for $name {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                serializer.serialize_bool($value)
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                let got = bool::deserialize(deserializer)?;
                if got == $value {
                    Ok($name)
                } else {
                    Err(D::Error::custom(concat!(
                        "expected the literal ",
                        stringify!($value)
                    )))
                }
            }
        }

        impl JsonSchema for $name {
            fn schema_name() -> Cow<'static, str> {
                Cow::Borrowed(stringify!($name))
            }

            fn json_schema(_gen: &mut SchemaGenerator) -> Schema {
                schemars::json_schema!({ "const": $value })
            }
        }
    };
}

literal_bool!(LiteralTrue, true);
literal_bool!(LiteralFalse, false);

macro_rules! literal_str {
    ($name:ident, $value:literal) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
        pub struct $name;

        impl $name {
            pub const VALUE: &'static str = $value;
        }

        impl Serialize for $name {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                serializer.serialize_str($value)
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                let got = String::deserialize(deserializer)?;
                if got == $value {
                    Ok($name)
                } else {
                    Err(D::Error::custom(concat!("expected the literal \"", $value, "\"")))
                }
            }
        }

        impl JsonSchema for $name {
            fn schema_name() -> Cow<'static, str> {
                Cow::Borrowed(stringify!($name))
            }

            fn json_schema(_gen: &mut SchemaGenerator) -> Schema {
                schemars::json_schema!({ "const": $value })
            }
        }
    };
}

literal_str!(ReloadedLiteral, "reloaded");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literal_bool_roundtrip() {
        assert_eq!(serde_json::to_string(&LiteralTrue).unwrap(), "true");
        assert!(serde_json::from_str::<LiteralTrue>("true").is_ok());
        assert!(serde_json::from_str::<LiteralTrue>("false").is_err());
        assert_eq!(serde_json::to_string(&LiteralFalse).unwrap(), "false");
        assert!(serde_json::from_str::<LiteralFalse>("false").is_ok());
        assert!(serde_json::from_str::<LiteralFalse>("true").is_err());
    }

    #[test]
    fn literal_str_roundtrip() {
        assert_eq!(
            serde_json::to_string(&ReloadedLiteral).unwrap(),
            "\"reloaded\""
        );
        assert!(serde_json::from_str::<ReloadedLiteral>("\"reloaded\"").is_ok());
        assert!(serde_json::from_str::<ReloadedLiteral>("\"nope\"").is_err());
    }
}
