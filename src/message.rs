use std::{sync::Arc, vec};

use agent_stream_kit::{AgentError, AgentValue};
use serde::{Deserialize, Serialize};

#[cfg(feature = "image")]
use photon_rs::PhotonImage;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,

    pub content: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    #[cfg(feature = "image")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<Arc<PhotonImage>>,
}

impl Message {
    pub fn new(role: String, content: String) -> Self {
        Self {
            role,
            content,
            id: None,

            #[cfg(feature = "image")]
            image: None,
        }
    }

    pub fn assistant(content: String) -> Self {
        Message::new("assistant".to_string(), content)
    }

    pub fn system(content: String) -> Self {
        Message::new("system".to_string(), content)
    }

    pub fn user(content: String) -> Self {
        Message::new("user".to_string(), content)
    }

    #[cfg(feature = "image")]
    pub fn with_image(mut self, image: Arc<PhotonImage>) -> Self {
        self.image = Some(image);
        self
    }
}

impl TryFrom<AgentValue> for Message {
    type Error = AgentError;

    fn try_from(value: AgentValue) -> Result<Self, Self::Error> {
        match value {
            AgentValue::String(s) => Ok(Message::user(s.to_string())),

            #[cfg(feature = "image")]
            AgentValue::Image(img) => {
                let mut message = Message::user("".to_string());
                message.image = Some(img.clone());
                Ok(message)
            }
            AgentValue::Object(obj) => {
                let role = obj
                    .get("role")
                    .and_then(|r| r.as_str())
                    .unwrap_or("user")
                    .to_string();
                let content = obj
                    .get("content")
                    .and_then(|c| c.as_str())
                    .ok_or_else(|| {
                        AgentError::InvalidValue(
                            "Message object missing 'content' field".to_string(),
                        )
                    })?
                    .to_string();
                let id = obj
                    .get("id")
                    .and_then(|i| i.as_str())
                    .map(|s| s.to_string());
                let mut message = Message::new(role, content);
                message.id = id;

                #[cfg(feature = "image")]
                {
                    if let Some(image_value) = obj.get("image") {
                        match image_value {
                            AgentValue::String(s) => {
                                message.image = Some(Arc::new(PhotonImage::new_from_base64(
                                    s.trim_start_matches("data:image/png;base64,"),
                                )));
                            }
                            AgentValue::Image(img) => {
                                message.image = Some(img.clone());
                            }
                            _ => {}
                        }
                    }
                }

                Ok(message)
            }
            _ => Err(AgentError::InvalidValue(
                "Cannot convert AgentValue to Message".to_string(),
            )),
        }
    }
}

impl From<Message> for AgentValue {
    fn from(msg: Message) -> Self {
        let mut fields = vec![
            ("role".to_string(), AgentValue::string(msg.role)),
            ("content".to_string(), AgentValue::string(msg.content)),
        ];
        if let Some(id_str) = msg.id {
            fields.push(("id".to_string(), AgentValue::string(id_str)));
        }
        #[cfg(feature = "image")]
        {
            if let Some(img) = msg.image {
                fields.push(("image".to_string(), AgentValue::image((*img).clone())));
            }
        }
        AgentValue::object(fields.into_iter().collect())
    }
}

#[derive(Clone, Default, Debug)]
pub struct MessageHistory {
    messages: Vec<Message>,
    max_size: usize,
    system_message: Option<Message>,
    include_system: bool,
}

impl MessageHistory {
    pub fn new(messages: Vec<Message>, max_size: usize) -> Self {
        let mut hist = Self {
            messages,
            max_size: 0,
            system_message: None,
            include_system: false,
        };
        hist.set_max_size(max_size);
        hist
    }

    /// Create MessageHistory from a JSON value
    pub fn from_json(value: serde_json::Value) -> Result<Self, AgentError> {
        match value {
            serde_json::Value::Array(arr) => {
                let messages: Vec<Message> = arr
                    .into_iter()
                    .map(|v| serde_json::from_value(v))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| {
                        AgentError::InvalidValue(format!("Invalid message format: {}", e))
                    })?;
                Ok(MessageHistory::new(messages, 0))
            }
            _ => Err(AgentError::InvalidValue(
                "Expected JSON array for MessageHistory".to_string(),
            )),
        }
    }

    /// Parse MessageHistory from a JSON string
    pub fn parse(s: &str) -> Result<Self, AgentError> {
        let value: serde_json::Value = serde_json::from_str(s).map_err(|e| {
            AgentError::InvalidValue(format!("Failed to parse JSON for MessageHistory: {}", e))
        })?;
        Self::from_json(value)
    }

    /// Get the messages in the history, including system message if configured.
    pub fn messages(&self) -> Vec<Message> {
        let mut msgs = Vec::new();
        if self.include_system {
            if let Some(sys_msg) = &self.system_message {
                msgs.push(sys_msg.clone());
            }
            msgs.extend(self.messages.clone());
            msgs
        } else {
            self.messages.clone()
        }
    }

    pub fn include_system(&self) -> bool {
        self.include_system
    }

    pub fn set_include_system(&mut self, include: bool) {
        self.include_system = include;
    }

    /// Set the maximum size of the message history.
    /// If max_size is 0, there is no limit.
    /// If the current size exceeds the new maximum, the oldest messages will be removed.
    /// If include_system is true, the system message will be preserved.
    pub fn set_max_size(&mut self, size: usize) {
        self.max_size = size;
        if self.max_size > 0 && self.messages.len() > self.max_size {
            if self.include_system {
                // find system message if it will be excluded from history
                for i in 0..(self.messages.len() - self.max_size) {
                    if self.messages[i].role == "system" {
                        self.system_message = Some(self.messages[i].clone());
                        break;
                    }
                }
            }
            self.messages = self.messages[self.messages.len() - self.max_size..].to_vec();
        }
    }

    pub fn set_preamble(&mut self, preamble: Vec<Message>) {
        if preamble.is_empty() {
            return;
        }
        let mut msgs = vec![];
        msgs.extend(preamble.clone());
        msgs.extend(self.messages.clone());
        self.messages = msgs;
        self.system_message = None;
        self.set_max_size(self.max_size);
    }

    /// Push a new message to the history.
    /// If the message has the same ID as the last message, it will update the last message instead.
    /// If the history exceeds max_size, the oldest message will be removed.
    /// If include_system is true and the removed message is a system message, it will be preserved.
    pub fn push(&mut self, message: Message) {
        // If the message is the same as the last one, update it instead of adding a new one
        if message.id.is_some() && !self.messages.is_empty() {
            let last_index = self.messages.len() - 1;
            let last_message = &mut self.messages[last_index];
            if last_message.id.is_some() && last_message.id == message.id {
                last_message.content = message.content;
                return;
            }
        }

        if self.max_size > 0 && self.messages.len() >= self.max_size {
            let m = self.messages.remove(0);
            if m.role == "system" {
                self.system_message = Some(m);
            }
        }
        self.messages.push(message);
    }
}

impl From<MessageHistory> for AgentValue {
    fn from(history: MessageHistory) -> Self {
        AgentValue::array(history.messages.into_iter().map(|m| m.into()).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Message tests

    #[test]
    fn test_message_to_from_agent_value() {
        let msg = Message::user("What is the weather today?".to_string());
        let value: AgentValue = msg.clone().into();
        let msg_converted: Message = value.try_into().unwrap();
        assert_eq!(msg_converted.role, "user");
        assert_eq!(msg_converted.content, "What is the weather today?");
    }

    #[test]
    fn test_message_from_string_value() {
        let value = AgentValue::string("Just a simple message");
        let msg: Message = value.try_into().unwrap();
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content, "Just a simple message");
    }

    #[test]
    fn test_message_from_object_value() {
        let value = AgentValue::object(
            [
                ("role".to_string(), AgentValue::string("assistant")),
                (
                    "content".to_string(),
                    AgentValue::string("Here is some information."),
                ),
            ]
            .into(),
        );
        let msg: Message = value.try_into().unwrap();
        assert_eq!(msg.role, "assistant");
        assert_eq!(msg.content, "Here is some information.");
    }

    #[test]
    fn test_message_from_invalid_value() {
        let value = AgentValue::integer(42);
        let result: Result<Message, AgentError> = value.try_into();
        assert!(result.is_err());
    }

    #[test]
    fn test_message_invalid_object() {
        let value =
            AgentValue::object([("some_key".to_string(), AgentValue::string("some_value"))].into());
        let result: Result<Message, AgentError> = value.try_into();
        assert!(result.is_err());
    }

    // MessageHistory tests

    const SAMPLE_HISTORY: &str = r#"
    [
        { "role": "system", "content": "You are a helpful assistant." },
        { "role": "user", "content": "Hello" },
        { "role": "assistant", "content": "Hi there!" }
    ]"#;

    #[test]
    fn test_message_history_new() {
        let history = MessageHistory::new(vec![], 0);
        assert_eq!(history.messages.len(), 0);
        assert_eq!(history.max_size, 0);
        assert_eq!(history.include_system, false);
        assert!(history.system_message.is_none());
    }

    #[test]
    fn test_message_history_from_json() {
        let value: serde_json::Value = serde_json::json!([
            { "role": "user", "content": "Hello" },
            { "role": "assistant", "content": "Hi there!" }
        ]);
        let history = MessageHistory::from_json(value).unwrap();
        assert_eq!(history.messages.len(), 2);
        assert_eq!(history.messages[0].role, "user");
        assert_eq!(history.messages[0].content, "Hello");
        assert_eq!(history.messages[1].role, "assistant");
        assert_eq!(history.messages[1].content, "Hi there!");
    }

    #[test]
    fn test_message_history_parse() {
        let history = MessageHistory::parse(SAMPLE_HISTORY).unwrap();
        assert_eq!(history.messages.len(), 3);
        assert_eq!(history.messages[0].role, "system");
        assert_eq!(history.messages[0].content, "You are a helpful assistant.");
        assert_eq!(history.messages[1].role, "user");
        assert_eq!(history.messages[1].content, "Hello");
        assert_eq!(history.messages[2].role, "assistant");
        assert_eq!(history.messages[2].content, "Hi there!");
    }

    #[test]
    fn test_message_history_include_system() {
        let mut history = MessageHistory::new(vec![], 0);
        assert_eq!(history.include_system(), false);
        history.set_include_system(true);
        assert_eq!(history.include_system(), true);
    }

    #[test]
    fn test_message_history_set_max_size() {
        let mut history = MessageHistory::parse(SAMPLE_HISTORY).unwrap();
        assert_eq!(history.max_size, 0);
        assert_eq!(history.messages.len(), 3);

        history.set_max_size(5);
        assert_eq!(history.max_size, 5);
        assert_eq!(history.messages.len(), 3);

        history.set_max_size(1);
        assert_eq!(history.max_size, 1);
        let msgs = history.messages();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, "assistant");
    }

    #[test]
    fn test_message_history_set_max_size_with_include_system() {
        let mut history = MessageHistory::parse(SAMPLE_HISTORY).unwrap();
        history.set_include_system(true);
        assert_eq!(history.max_size, 0);
        assert_eq!(history.messages.len(), 3);

        history.set_max_size(5);
        assert_eq!(history.max_size, 5);
        assert_eq!(history.messages.len(), 3);

        history.set_max_size(1);
        assert_eq!(history.max_size, 1);
        assert_eq!(history.messages.len(), 1);
        assert!(history.system_message.is_some());
        let msgs = history.messages();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, "system");
        assert_eq!(msgs[1].role, "assistant");
    }

    #[test]
    fn test_message_history_push() {
        let mut history = MessageHistory::parse(SAMPLE_HISTORY).unwrap();
        let new_msg = Message::user("How are you?".to_string());
        history.push(new_msg);
        let msgs = history.messages();
        assert_eq!(msgs.len(), 4);
        assert_eq!(msgs[3].role, "user");
        assert_eq!(msgs[3].content, "How are you?");
    }

    #[test]
    fn test_message_history_push_with_include_system() {
        let mut history = MessageHistory::parse(SAMPLE_HISTORY).unwrap();
        history.set_include_system(true);
        history.set_max_size(3);
        assert_eq!(history.messages.len(), 3);
        assert!(history.system_message.is_none());

        let new_msg = Message::user("How are you?".to_string());
        history.push(new_msg);
        assert_eq!(history.messages.len(), 3);
        assert!(history.system_message.is_some());

        let msgs = history.messages();
        assert_eq!(msgs.len(), 4);
        assert_eq!(msgs[0].role, "system");
        assert_eq!(msgs[3].role, "user");

        let new_msg = Message::assistant("Good!".to_string());
        history.push(new_msg);
        assert_eq!(history.messages.len(), 3);
        assert!(history.system_message.is_some());

        let msgs = history.messages();
        assert_eq!(msgs.len(), 4);
        assert_eq!(msgs[0].role, "system");
        assert_eq!(msgs[3].role, "assistant");
    }

    #[test]
    fn test_message_history_push_update_last() {
        let mut history =
            MessageHistory::parse(r#"[{"role": "user", "content": "Hello", "id": "msg1"}]"#)
                .unwrap();
        let updated_msg = Message {
            role: "user".to_string(),
            content: "Hello, updated!".to_string(),
            id: Some("msg1".to_string()),
            #[cfg(feature = "image")]
            image: None,
        };
        history.push(updated_msg);
        let msgs = history.messages();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].content, "Hello, updated!");
    }
}
