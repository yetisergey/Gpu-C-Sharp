namespace gpu
{
    using OpenCLTemplate;
    using System;
    using System.Diagnostics;
    using System.IO;
    using System.Linq;
    class Program
    {
        static void Main()
        {
            while (true)
            {
                Stopwatch sw = new Stopwatch();
                sw.Start();
                GPU();
                sw.Stop();
                Stopwatch sw1 = new Stopwatch();
                sw1.Start();
                Processor();
                sw1.Stop();
                PrintSW(sw, sw1);
            }

        }
        private static void Processor()
        {
            int n = 20000000;
            float[] v1 = new float[n], v2 = new float[n];
            for (int i = 0; i < n; i++)
            {
                v1[i] = (float)i / 10;
                v2[i] = -(float)i / 8;
            }
            for (int i = 0; i < n; i++)
            {
                v1[i] = v1[i] + v2[i];
            }
        }
        private static void GPU()
        {
            string vecSum = @"
                     __kernel void
                    floatVectorSum(__global       float * v1,
                                   __global       float * v2)
                    {
                        // Vector element index
                        int i = get_global_id(0);
                        v1[i] = v1[i] + v2[i];
                    }";
            //Initializes OpenCL Platforms and Devices and sets everything up
            CLCalc.InitCL();
            //Compiles the source codes. The source is a string array because the user may want
            //to split the source into many strings.
            CLCalc.Program.Compile(new string[] { vecSum });
            //Gets host access to the OpenCL floatVectorSum kernel
            CLCalc.Program.Kernel VectorSum = new CLCalc.Program.Kernel("floatVectorSum");
            //We want to sum 2000 numbers
            int n = 20000000;
            //Create vectors with 2000 numbers
            float[] v1 = new float[n], v2 = new float[n];
            //Creates population for v1 and v2
            for (int i = 0; i < n; i++)
            {
                v1[i] = (float)i / 10;
                v2[i] = -(float)i / 8;
            }
            //Creates vectors v1 and v2 in the device memory
            CLCalc.Program.Variable varV1 = new CLCalc.Program.Variable(v1);
            CLCalc.Program.Variable varV2 = new CLCalc.Program.Variable(v2);
            //Arguments of VectorSum kernel
            CLCalc.Program.Variable[] args = new CLCalc.Program.Variable[] { varV1, varV2 };
            //How many workers will there be? We need "n", one for each element
            //int[] workers = new int[1] { n };
            //Execute the kernel
            VectorSum.Execute(args, 20000000);
            //Read device memory varV1 to host memory v1
            varV1.ReadFromDeviceTo(v1);
        }
        private static void PrintSW(Stopwatch sw, Stopwatch sw1)
        {
            using (var swr = new StreamWriter("out.txt",true))
            {
                TimeSpan ts = sw.Elapsed;
                string elapsedTime = string.Format("{0:00}:{1:00}:{2:00}.{3:00}",
                    ts.Hours, ts.Minutes, ts.Seconds,
                    ts.Milliseconds / 10);
                TimeSpan ts1 = sw1.Elapsed;
                string elapsedTime1 = string.Format("{0:00}:{1:00}:{2:00}.{3:00}",
                    ts1.Hours, ts1.Minutes, ts1.Seconds,
                    ts1.Milliseconds / 10);
                swr.WriteLine("//===============================");
                swr.WriteLine("GPU  =  " + elapsedTime);
                swr.WriteLine("CPU  =  " + elapsedTime1);
                swr.WriteLine("//===============================");
            }
        }
    }
}