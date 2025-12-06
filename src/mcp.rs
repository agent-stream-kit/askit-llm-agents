#![cfg(feature = "mcp")]

use std::vec;

use agent_stream_kit::{
    ASKit, Agent, AgentConfigs, AgentContext, AgentData, AgentError, AgentOutput, AgentValue,
    AsAgent, async_trait,
};
use askit_macros::askit_agent;
use rmcp::{
    model::{CallToolRequestParam, CallToolResult},
    service::ServiceExt,
    transport::{ConfigureCommandExt, TokioChildProcess},
};
use tokio::process::Command;

static CATEGORY: &str = "LLM/MCP";

static PIN_UNIT: &str = "unit";
static PIN_VALUE: &str = "value";
static PIN_RESPONSE: &str = "response";

static CONFIG_COMMAND: &str = "command";
static CONFIG_ARGS: &str = "args";
static CONFIG_TOOL: &str = "tool";

#[askit_agent(
    title="MCP Tools List",
    category=CATEGORY,
    inputs=[PIN_UNIT],
    outputs=[PIN_VALUE],
    string_config(name=CONFIG_COMMAND),
    string_config(name=CONFIG_ARGS),
)]
pub struct MCPToolsListAgent {
    data: AgentData,
}

#[async_trait]
impl AsAgent for MCPToolsListAgent {
    fn new(
        askit: ASKit,
        id: String,
        def_name: String,
        config: Option<AgentConfigs>,
    ) -> Result<Self, AgentError> {
        Ok(Self {
            data: AgentData::new(askit, id, def_name, config),
        })
    }
    async fn process(
        &mut self,
        ctx: AgentContext,
        _pin: String,
        _value: AgentValue,
    ) -> Result<(), AgentError> {
        let command = self.configs()?.get_string_or_default(CONFIG_COMMAND);
        let args_str = self.configs()?.get_string_or_default(CONFIG_ARGS);
        let args: Vec<String> = serde_json::from_str(&args_str)
            .map_err(|e| AgentError::InvalidValue(format!("Failed to parse args JSON: {e}")))?;

        let service = ()
            .serve(
                TokioChildProcess::new(Command::new(&command).configure(|cmd| {
                    for arg in &args {
                        cmd.arg(arg);
                    }
                }))
                .map_err(|e| AgentError::Other(format!("Failed to start MCP process: {e}")))?,
            )
            .await
            .map_err(|e| AgentError::Other(format!("Failed to start MCP service: {e}")))?;

        let tools_list = service
            .list_tools(Default::default())
            .await
            .map_err(|e| AgentError::Other(format!("Failed to list MCP tools: {e}")))?;

        service
            .cancel()
            .await
            .map_err(|e| AgentError::Other(format!("Failed to cancel MCP service: {e}")))?;

        let tools_value = AgentValue::from_serialize(&tools_list).map_err(|e| {
            AgentError::Other(format!(
                "Failed to serialize MCP tools list to AgentValue: {e}"
            ))
        })?;

        self.try_output(ctx, PIN_VALUE, tools_value)?;

        Ok(())
    }
}

#[askit_agent(
    title="MCP Call",
    category=CATEGORY,
    inputs=[PIN_VALUE],
    outputs=[PIN_VALUE, PIN_RESPONSE],
    string_config(name=CONFIG_COMMAND),
    string_config(name=CONFIG_ARGS),
    string_config(name=CONFIG_TOOL),
)]
pub struct MCPCallAgent {
    data: AgentData,
}

#[async_trait]
impl AsAgent for MCPCallAgent {
    fn new(
        askit: ASKit,
        id: String,
        def_name: String,
        config: Option<AgentConfigs>,
    ) -> Result<Self, AgentError> {
        Ok(Self {
            data: AgentData::new(askit, id, def_name, config),
        })
    }

    async fn process(
        &mut self,
        ctx: AgentContext,
        _pin: String,
        value: AgentValue,
    ) -> Result<(), AgentError> {
        let command = self.configs()?.get_string_or_default(CONFIG_COMMAND);
        let args_str = self.configs()?.get_string_or_default(CONFIG_ARGS);
        let args: Vec<String> = serde_json::from_str(&args_str)
            .map_err(|e| AgentError::InvalidValue(format!("Failed to parse args JSON: {e}")))?;

        let service = ()
            .serve(
                TokioChildProcess::new(Command::new(&command).configure(|cmd| {
                    for arg in &args {
                        cmd.arg(arg);
                    }
                }))
                .map_err(|e| AgentError::Other(format!("Failed to start MCP process: {e}")))?,
            )
            .await
            .map_err(|e| AgentError::Other(format!("Failed to start MCP service: {e}")))?;

        let tool_name = self.configs()?.get_string_or_default(CONFIG_TOOL);
        if tool_name.is_empty() {
            return Ok(());
        }

        let arguments = value.as_object().map(|obj| {
            obj.iter()
                .map(|(k, v)| {
                    (
                        k.clone(),
                        serde_json::to_value(v).unwrap_or(serde_json::Value::Null),
                    )
                })
                .collect::<serde_json::Map<String, serde_json::Value>>()
        });

        let tool_result = service
            .call_tool(CallToolRequestParam {
                name: tool_name.clone().into(),
                arguments,
            })
            .await
            .map_err(|e| AgentError::Other(format!("Failed to call tool '{}': {e}", tool_name)))?;

        service
            .cancel()
            .await
            .map_err(|e| AgentError::Other(format!("Failed to cancel MCP service: {e}")))?;

        self.try_output(
            ctx.clone(),
            PIN_VALUE,
            call_tool_result_to_agent_value(tool_result.clone())?,
        )?;

        let response = serde_json::to_string_pretty(&tool_result).map_err(|e| {
            AgentError::Other(format!(
                "Failed to serialize tool result content to JSON: {e}"
            ))
        })?;
        self.try_output(ctx, PIN_RESPONSE, AgentValue::string(response))?;

        Ok(())
    }
}

fn call_tool_result_to_agent_value(result: CallToolResult) -> Result<AgentValue, AgentError> {
    let mut contents = Vec::new();
    for c in result.content.iter() {
        match &c.raw {
            rmcp::model::RawContent::Text(text) => {
                contents.push(AgentValue::string(text.text.clone()));
            }
            _ => {
                // Handle other content types as needed
            }
        }
    }
    let data = AgentValue::array(contents);
    if result.is_error == Some(true) {
        return Err(AgentError::Other(
            serde_json::to_string(&data).map_err(|e| AgentError::InvalidValue(e.to_string()))?,
        ));
    }
    Ok(data)
}
