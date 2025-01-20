from enum import Enum, auto

class Language(Enum):
    """Supported programming languages for code analysis"""
    PYTHON = "python"
    JAVASCRIPT = "javascript" 
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    C = "c"

    @property
    def extension(self) -> str:
        """Get the file extension for this language"""
        extensions = {
            Language.PYTHON: ".py",
            Language.JAVASCRIPT: ".js",
            Language.TYPESCRIPT: ".ts", 
            Language.JAVA: ".java",
            Language.CPP: ".cpp",
            Language.C: ".c"
        }
        return extensions[self]
