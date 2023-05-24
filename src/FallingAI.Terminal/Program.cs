using CommandDotNet;

namespace FallingAI.Terminal
{
    internal static class Program
    {

        private static int Main(string[] args)
        {
            return new AppRunner<ConsoleCommands>().Run(args);
        }
    }
}
