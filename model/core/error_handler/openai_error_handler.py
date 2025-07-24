# # model/core/error_handlers/openai_error_handler.py
#
# import openai
#
# from model.core.error_handler.llm_error_handler import LLMErrorHandler
#
#
# class OpenAIErrorHandler(LLMErrorHandler):
#     @property
#     def retryable_errors(self):
#         return (
#             openai.RateLimitError,
#             openai.APIConnectionError,
#             openai.Timeout,
#         )
#
#     @property
#     def user_errors(self):
#         return (
#             openai.AuthenticationError,
#             openai.PermissionDeniedError,
#             openai.InvalidRequestError,
#         )
